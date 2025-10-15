import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import logging
from datetime import datetime
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import yaml

class DataIngestion:
    """Clase para ingesta y validación inicial de datos."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Inicializar ingesta de datos.
        
        Args:
            config_path: Ruta al archivo de configuración
        """
        self.logger = logging.getLogger(__name__)
        self.config = self._load_config(config_path) if config_path else {}
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Cargar configuración desde archivo YAML."""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            self.logger.warning(f"No se pudo cargar configuración: {e}")
            return {}
    
    def load_loan_data(self, file_path: str, **kwargs) -> pd.DataFrame:
        """
        Cargar datos de préstamos desde CSV.
        
        Args:
            file_path: Ruta al archivo CSV
            **kwargs: Argumentos adicionales para pd.read_csv
            
        Returns:
            DataFrame con datos cargados
        """
        try:
            # Parámetros por defecto optimizados para datos de préstamos
            default_params = {
                'encoding': 'utf-8',
                'low_memory': False,
                'parse_dates': ['issue_d', 'earliest_cr_line', 'last_pymnt_d', 'next_pymnt_d'],
                'na_values': ['n/a', 'N/A', 'NULL', 'null', '']
            }
            default_params.update(kwargs)
            
            df = pd.read_csv(file_path, **default_params)
            
            self.logger.info(f"Datos cargados exitosamente: {df.shape[0]} filas, {df.shape[1]} columnas")
            
            # Validación básica
            self._validate_basic_structure(df)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error cargando datos: {e}")
            raise
    
    def _validate_basic_structure(self, df: pd.DataFrame) -> None:
        """Validar estructura básica del dataset."""
        
        # Verificar columnas críticas
        critical_columns = ['loan_amnt', 'funded_amnt', 'int_rate', 'grade', 'loan_status']
        missing_critical = [col for col in critical_columns if col not in df.columns]
        
        if missing_critical:
            raise ValueError(f"Columnas críticas faltantes: {missing_critical}")
        
        # Verificar datos vacíos
        if df.empty:
            raise ValueError("Dataset está vacío")
        
        # Log estadísticas básicas
        self.logger.info(f"Columnas en dataset: {list(df.columns[:10])}{'...' if len(df.columns) > 10 else ''}")
        self.logger.info(f"Tipos de datos únicos: {df.dtypes.value_counts().to_dict()}")
        self.logger.info(f"Valores nulos por columna (top 5): {df.isnull().sum().nlargest(5).to_dict()}")

class DataCleaning:
    """Clase para limpieza y preprocesamiento de datos."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Inicializar limpieza de datos.
        
        Args:
            config: Configuración para limpieza
        """
        self.logger = logging.getLogger(__name__)
        self.config = config or {}
        self.encoders = {}
        self.scalers = {}
        
    def clean_loan_status(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Limpiar y estandarizar loan_status.
        
        Args:
            df: DataFrame original
            
        Returns:
            DataFrame con loan_status limpio
        """
        df = df.copy()
        
        # Mapeo estándar de loan_status
        status_mapping = {
            'Fully Paid': 'Paid',
            'Current': 'Current',
            'Charged Off': 'Default',
            'Default': 'Default',
            'Late (31-120 days)': 'Late',
            'Late (16-30 days)': 'Late',
            'In Grace Period': 'Current',
            'Issued': 'Current'
        }
        
        # Aplicar mapeo
        df['loan_status_clean'] = df['loan_status'].map(status_mapping)
        
        # Manejar valores no mapeados
        unmapped = df[df['loan_status_clean'].isnull()]['loan_status'].unique()
        if len(unmapped) > 0:
            self.logger.warning(f"Valores de loan_status no mapeados: {unmapped}")
            df['loan_status_clean'] = df['loan_status_clean'].fillna('Other')
        
        return df
    
    def create_target_variables(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Crear variables target para modelos PD, LGD, EAD.
        
        Args:
            df: DataFrame con datos limpios
            
        Returns:
            DataFrame con variables target
        """
        df = df.copy()
        
        # Variable target para PD (Probability of Default)
        df['default_flag'] = (df['loan_status_clean'] == 'Default').astype(int)
        
        # Variable target para LGD (Loss Given Default) - solo para defaults
        if all(col in df.columns for col in ['funded_amnt', 'total_rec_prncp', 'recoveries']):
            # Pérdida = Monto financiado - Principal recuperado - Recuperaciones
            df['loss_amount'] = df['funded_amnt'] - df['total_rec_prncp'] - df['recoveries']
            df['loss_rate'] = df['loss_amount'] / df['funded_amnt']
            
            # LGD solo para préstamos en default
            df.loc[df['default_flag'] == 0, 'loss_rate'] = np.nan
            
            # Limitar LGD entre 0 y 1
            df['loss_rate'] = df['loss_rate'].clip(0, 1)
        
        # Variable target para EAD (Exposure at Default)
        if 'out_prncp' in df.columns:
            df['ead_amount'] = df['out_prncp']  # Saldo pendiente al momento del análisis
            if 'funded_amnt' in df.columns:
                df['ead_ratio'] = df['ead_amount'] / df['funded_amnt']
        
        return df
    
    def handle_missing_values(self, df: pd.DataFrame, 
                             strategy: Dict[str, str] = None) -> pd.DataFrame:
        """
        Manejar valores faltantes según estrategia definida.
        
        Args:
            df: DataFrame original
            strategy: Estrategia por tipo de columna
            
        Returns:
            DataFrame sin valores faltantes críticos
        """
        df = df.copy()
        
        if strategy is None:
            strategy = {
                'numerical': 'median',
                'categorical': 'mode',
                'drop_threshold': 0.5  # Eliminar columnas con >50% faltantes
            }
        
        # Eliminar columnas con muchos valores faltantes
        threshold = strategy.get('drop_threshold', 0.5)
        missing_pct = df.isnull().sum() / len(df)
        cols_to_drop = missing_pct[missing_pct > threshold].index.tolist()
        
        if cols_to_drop:
            self.logger.info(f"Eliminando columnas con >{threshold*100}% faltantes: {cols_to_drop}")
            df = df.drop(columns=cols_to_drop)
        
        # Manejar valores faltantes por tipo
        for column in df.columns:
            if df[column].isnull().sum() > 0:
                if df[column].dtype in ['object', 'category']:
                    # Categóricas
                    if strategy['categorical'] == 'mode':
                        mode_val = df[column].mode().iloc[0] if not df[column].mode().empty else 'Unknown'
                        df[column] = df[column].fillna(mode_val)
                    elif strategy['categorical'] == 'unknown':
                        df[column] = df[column].fillna('Unknown')
                        
                else:
                    # Numéricas
                    if strategy['numerical'] == 'median':
                        df[column] = df[column].fillna(df[column].median())
                    elif strategy['numerical'] == 'mean':
                        df[column] = df[column].fillna(df[column].mean())
                    elif strategy['numerical'] == 'zero':
                        df[column] = df[column].fillna(0)
        
        return df
    
    def handle_outliers(self, df: pd.DataFrame, columns: List[str] = None,
                       method: str = 'iqr', threshold: float = 1.5) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Detectar y manejar outliers.
        
        Args:
            df: DataFrame original
            columns: Columnas a analizar (por defecto numéricas)
            method: Método de detección ('iqr', 'zscore', 'isolation_forest')
            threshold: Umbral para detección
            
        Returns:
            Tupla (DataFrame limpio, DataFrame con outliers)
        """
        df = df.copy()
        
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        outlier_mask = pd.Series(False, index=df.index)
        
        for column in columns:
            if column in df.columns:
                if method == 'iqr':
                    Q1 = df[column].quantile(0.25)
                    Q3 = df[column].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - threshold * IQR
                    upper_bound = Q3 + threshold * IQR
                    
                    column_outliers = (df[column] < lower_bound) | (df[column] > upper_bound)
                    
                elif method == 'zscore':
                    z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
                    column_outliers = z_scores > threshold
                
                outlier_mask |= column_outliers
        
        df_clean = df[~outlier_mask].copy()
        df_outliers = df[outlier_mask].copy()
        
        self.logger.info(f"Outliers detectados: {outlier_mask.sum()} ({outlier_mask.mean()*100:.1f}%)")
        
        return df_clean, df_outliers
    
    def encode_categorical_features(self, df: pd.DataFrame, 
                                  categorical_columns: List[str] = None) -> pd.DataFrame:
        """
        Codificar variables categóricas.
        
        Args:
            df: DataFrame original
            categorical_columns: Columnas categóricas a codificar
            
        Returns:
            DataFrame con variables codificadas
        """
        df = df.copy()
        
        if categorical_columns is None:
            categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        for column in categorical_columns:
            if column in df.columns:
                # Usar LabelEncoder para variables categóricas
                if column not in self.encoders:
                    self.encoders[column] = LabelEncoder()
                    df[f'{column}_encoded'] = self.encoders[column].fit_transform(df[column].astype(str))
                else:
                    # Usar encoder previamente entrenado
                    df[f'{column}_encoded'] = self.encoders[column].transform(df[column].astype(str))
        
        return df
    
    def create_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Crear features derivadas específicas para riesgo crediticio.
        
        Args:
            df: DataFrame original
            
        Returns:
            DataFrame con features derivadas
        """
        df = df.copy()
        
        try:
            # Ratio deuda-ingreso ajustado
            if all(col in df.columns for col in ['installment', 'annual_inc']):
                df['dti_installment'] = (df['installment'] * 12) / df['annual_inc']
            
            # Utilización de crédito
            if all(col in df.columns for col in ['revol_bal', 'revol_util']):
                df['credit_utilization_amount'] = df['revol_bal']
            
            # Experiencia crediticia (años)
            if 'earliest_cr_line' in df.columns:
                df['credit_history_length'] = (pd.Timestamp.now() - df['earliest_cr_line']).dt.days / 365.25
            
            # Ratio monto solicitado vs ingresos
            if all(col in df.columns for col in ['loan_amnt', 'annual_inc']):
                df['loan_to_income'] = df['loan_amnt'] / df['annual_inc']
            
            # FICO promedio
            if all(col in df.columns for col in ['fico_range_low', 'fico_range_high']):
                df['fico_avg'] = (df['fico_range_low'] + df['fico_range_high']) / 2
            
            # Indicador de alto riesgo
            if 'grade' in df.columns:
                high_risk_grades = ['E', 'F', 'G']
                df['high_risk_grade'] = df['grade'].isin(high_risk_grades).astype(int)
            
            # Ratio de pagos realizados (para LGD)
            if all(col in df.columns for col in ['total_pymnt', 'funded_amnt']):
                df['payment_ratio'] = df['total_pymnt'] / df['funded_amnt']
            
            self.logger.info(f"Features derivadas creadas: {[col for col in df.columns if col.endswith(('_ratio', '_length', '_avg', '_amount', '_installment'))]}")
            
        except Exception as e:
            self.logger.warning(f"Error creando features derivadas: {e}")
        
        return df

class FeatureSelector:
    """Clase para selección de features."""
    
    def __init__(self, target_variable: str):
        """
        Inicializar selector de features.
        
        Args:
            target_variable: Variable objetivo
        """
        self.target_variable = target_variable
        self.logger = logging.getLogger(__name__)
        self.selected_features = []
    
    def select_features_by_correlation(self, df: pd.DataFrame, 
                                     threshold: float = 0.95) -> List[str]:
        """
        Seleccionar features eliminando alta correlación.
        
        Args:
            df: DataFrame con features
            threshold: Umbral de correlación
            
        Returns:
            Lista de features seleccionadas
        """
        # Solo features numéricas
        numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if self.target_variable in numeric_features:
            numeric_features.remove(self.target_variable)
        
        # Calcular matriz de correlación
        corr_matrix = df[numeric_features].corr().abs()
        
        # Encontrar pares con alta correlación
        upper_triangle = corr_matrix.where(
            np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
        )
        
        # Features a eliminar
        to_drop = [column for column in upper_triangle.columns 
                  if any(upper_triangle[column] > threshold)]
        
        selected_features = [col for col in numeric_features if col not in to_drop]
        
        self.logger.info(f"Features eliminadas por alta correlación (>{threshold}): {len(to_drop)}")
        self.logger.info(f"Features seleccionadas: {len(selected_features)}")
        
        return selected_features
    
    def select_features_by_importance(self, df: pd.DataFrame, 
                                    method: str = 'mutual_info',
                                    top_k: int = 50) -> List[str]:
        """
        Seleccionar features por importancia.
        
        Args:
            df: DataFrame con features
            method: Método de selección ('mutual_info', 'chi2', 'f_classif')
            top_k: Número de features a seleccionar
            
        Returns:
            Lista de features seleccionadas
        """
        from sklearn.feature_selection import mutual_info_classif, SelectKBest, chi2, f_classif
        
        # Preparar datos
        numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
        if self.target_variable in numeric_features:
            numeric_features.remove(self.target_variable)
        
        X = df[numeric_features].fillna(0)
        y = df[self.target_variable]
        
        # Seleccionar método
        if method == 'mutual_info':
            selector = SelectKBest(mutual_info_classif, k=min(top_k, len(numeric_features)))
        elif method == 'chi2':
            # Chi2 requiere valores no negativos
            X = X - X.min() + 1
            selector = SelectKBest(chi2, k=min(top_k, len(numeric_features)))
        elif method == 'f_classif':
            selector = SelectKBest(f_classif, k=min(top_k, len(numeric_features)))
        
        # Aplicar selección
        selector.fit(X, y)
        selected_features = [numeric_features[i] for i in selector.get_support(indices=True)]
        
        self.logger.info(f"Features seleccionadas por {method}: {len(selected_features)}")
        
        return selected_features