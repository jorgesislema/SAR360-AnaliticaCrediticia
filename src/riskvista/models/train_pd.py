import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb
import joblib
from pathlib import Path
import yaml
from typing import Dict, Any, Tuple, Optional
import logging
import shap
from datetime import datetime

from ..utils.reporting import setup_logging, write_report, log_operation, ModelValidator
from ..utils.data_processing import DataIngestion, DataCleaning, FeatureSelector

class PDModel:
    """Modelo de Probabilidad de Default (PD)."""
    
    def __init__(self, config_path: str = "configs/pd.yaml"):
        """
        Inicializar modelo PD.
        
        Args:
            config_path: Ruta al archivo de configuración
        """
        self.logger = setup_logging()
        self.config = self._load_config(config_path)
        self.model = None
        self.feature_columns = []
        self.scaler = StandardScaler()
        self.validator = ModelValidator("pd")
        
        # Métricas de entrenamiento
        self.training_metrics = {}
        self.feature_importance = {}
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Cargar configuración del modelo."""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            self.logger.error(f"Error cargando configuración: {e}")
            return {}
    
    def prepare_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Preparar datos para entrenamiento del modelo PD.
        
        Args:
            df: DataFrame con datos de préstamos
            
        Returns:
            Tupla (features, target)
        """
        log_operation("Iniciando preparación de datos para modelo PD")
        
        # Limpiar datos
        cleaner = DataCleaning(self.config.get('data', {}))
        df_clean = cleaner.clean_loan_status(df)
        df_clean = cleaner.create_target_variables(df_clean)
        df_clean = cleaner.handle_missing_values(df_clean)
        df_clean = cleaner.create_derived_features(df_clean)
        
        # Seleccionar features
        categorical_features = self.config.get('features', {}).get('categorical_features', [])
        numerical_features = self.config.get('features', {}).get('numerical_features', [])
        
        # Codificar categóricas
        if categorical_features:
            df_clean = cleaner.encode_categorical_features(df_clean, categorical_features)
            categorical_encoded = [f"{col}_encoded" for col in categorical_features if col in df.columns]
        else:
            categorical_encoded = []
        
        # Combinar features
        all_features = []
        for feature in numerical_features:
            if feature in df_clean.columns:
                all_features.append(feature)
        all_features.extend(categorical_encoded)
        
        # Agregar features derivadas disponibles
        derived_features = [col for col in df_clean.columns 
                          if col.endswith(('_ratio', '_length', '_avg', '_amount', '_installment'))]
        all_features.extend(derived_features)
        
        # Remover duplicados y features que no existen
        self.feature_columns = [col for col in list(set(all_features)) if col in df_clean.columns]
        
        # Variable target
        target_var = self.config.get('features', {}).get('target_variable', 'default_flag')
        
        if target_var not in df_clean.columns:
            raise ValueError(f"Variable target '{target_var}' no encontrada en los datos")
        
        # Filtrar filas con target válido
        df_clean = df_clean.dropna(subset=[target_var])
        
        X = df_clean[self.feature_columns].copy()
        y = df_clean[target_var].copy()
        
        log_operation(f"Datos preparados: {X.shape[0]} filas, {X.shape[1]} features")
        log_operation(f"Distribución target: {y.value_counts().to_dict()}")
        
        return X, y
    
    def train(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """
        Entrenar modelo PD.
        
        Args:
            X: Features
            y: Variable target
            
        Returns:
            Métricas de entrenamiento
        """
        log_operation("Iniciando entrenamiento modelo PD")
        
        # Dividir datos
        data_config = self.config.get('data', {})
        test_size = data_config.get('test_size', 0.15)
        val_size = data_config.get('val_size', 0.15)
        random_state = data_config.get('random_state', 42)
        
        # Split inicial
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Split train/validation
        val_size_adj = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adj, random_state=random_state, stratify=y_temp
        )
        
        # Escalar features numéricas
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Configurar modelo
        model_config = self.config.get('model', {})
        algorithm = model_config.get('algorithm', 'xgboost')
        hyperparams = model_config.get('hyperparameters', {})
        
        if algorithm == 'xgboost':
            self.model = xgb.XGBClassifier(**hyperparams)
            # Entrenar con early stopping
            self.model.fit(
                X_train_scaled, y_train,
                eval_set=[(X_val_scaled, y_val)],
                verbose=False
            )
            
        elif algorithm == 'lightgbm':
            self.model = lgb.LGBMClassifier(**hyperparams)
            self.model.fit(
                X_train_scaled, y_train,
                eval_set=[(X_val_scaled, y_val)],
                callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
            )
        
        # Predicciones
        y_train_pred = self.model.predict_proba(X_train_scaled)[:, 1]
        y_val_pred = self.model.predict_proba(X_val_scaled)[:, 1]
        y_test_pred = self.model.predict_proba(X_test_scaled)[:, 1]
        
        # Calcular métricas
        metrics = self._calculate_metrics(
            {
                'train': (y_train, y_train_pred),
                'validation': (y_val, y_val_pred),
                'test': (y_test, y_test_pred)
            }
        )
        
        # Feature importance
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance = dict(zip(self.feature_columns, self.model.feature_importances_))
        
        # Cross-validation
        cv_config = self.config.get('model', {}).get('cv', {})
        if cv_config.get('n_splits', 0) > 0:
            cv_scores = self._cross_validate(X, y)
            metrics['cross_validation'] = cv_scores
        
        self.training_metrics = metrics
        
        log_operation(f"Entrenamiento completado. AUC Test: {metrics['test']['roc_auc']:.4f}")
        
        return metrics
    
    def _calculate_metrics(self, predictions: Dict[str, Tuple]) -> Dict[str, Any]:
        """Calcular métricas de evaluación."""
        from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, log_loss
        
        metrics = {}
        
        for split_name, (y_true, y_pred_proba) in predictions.items():
            y_pred_binary = (y_pred_proba > 0.5).astype(int)
            
            split_metrics = {
                'roc_auc': roc_auc_score(y_true, y_pred_proba),
                'precision': precision_score(y_true, y_pred_binary, zero_division=0),
                'recall': recall_score(y_true, y_pred_binary, zero_division=0),
                'f1_score': f1_score(y_true, y_pred_binary, zero_division=0),
                'accuracy': accuracy_score(y_true, y_pred_binary),
                'log_loss': log_loss(y_true, y_pred_proba)
            }
            
            # Métricas específicas de riesgo
            split_metrics.update(self._calculate_risk_metrics(y_true, y_pred_proba))
            
            metrics[split_name] = split_metrics
        
        return metrics
    
    def _calculate_risk_metrics(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> Dict[str, float]:
        """Calcular métricas específicas de riesgo crediticio."""
        from scipy.stats import ks_2samp
        
        # Coeficiente de Gini
        gini = 2 * roc_auc_score(y_true, y_pred_proba) - 1
        
        # Estadístico KS
        defaults = y_pred_proba[y_true == 1]
        non_defaults = y_pred_proba[y_true == 0]
        ks_stat, _ = ks_2samp(defaults, non_defaults)
        
        # Information Value (simplificado)
        n_bins = 10
        df_iv = pd.DataFrame({'target': y_true, 'score': y_pred_proba})
        df_iv['score_bin'] = pd.qcut(df_iv['score'], n_bins, duplicates='drop')
        
        iv_table = df_iv.groupby('score_bin')['target'].agg(['count', 'sum']).reset_index()
        iv_table['non_default'] = iv_table['count'] - iv_table['sum']
        
        total_default = iv_table['sum'].sum()
        total_non_default = iv_table['non_default'].sum()
        
        iv_table['default_rate'] = iv_table['sum'] / total_default
        iv_table['non_default_rate'] = iv_table['non_default'] / total_non_default
        
        # Evitar división por cero
        iv_table['woe'] = np.log((iv_table['non_default_rate'] + 1e-6) / (iv_table['default_rate'] + 1e-6))
        iv_table['iv'] = (iv_table['non_default_rate'] - iv_table['default_rate']) * iv_table['woe']
        
        information_value = iv_table['iv'].sum()
        
        return {
            'gini_coefficient': gini,
            'ks_statistic': ks_stat,
            'information_value': information_value
        }
    
    def _cross_validate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Realizar validación cruzada."""
        cv_config = self.config.get('model', {}).get('cv', {})
        n_splits = cv_config.get('n_splits', 5)
        
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        
        # Escalar datos completos
        X_scaled = self.scaler.fit_transform(X)
        
        # Configurar modelo para CV
        model_config = self.config.get('model', {})
        hyperparams = model_config.get('hyperparameters', {})
        
        if model_config.get('algorithm', 'xgboost') == 'xgboost':
            cv_model = xgb.XGBClassifier(**hyperparams)
        else:
            cv_model = lgb.LGBMClassifier(**hyperparams)
        
        # Realizar CV
        cv_scores = cross_val_score(cv_model, X_scaled, y, cv=skf, scoring='roc_auc', n_jobs=-1)
        
        return {
            'mean_cv_auc': cv_scores.mean(),
            'std_cv_auc': cv_scores.std(),
            'cv_scores': cv_scores.tolist()
        }
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Realizar predicciones.
        
        Args:
            X: Features para predicción
            
        Returns:
            Probabilidades de default
        """
        if self.model is None:
            raise ValueError("Modelo no entrenado. Ejecutar train() primero.")
        
        # Escalar features
        X_scaled = self.scaler.transform(X[self.feature_columns])
        
        # Predicciones
        predictions = self.model.predict_proba(X_scaled)[:, 1]
        
        # Validar predicciones
        validation_results = self.validator.validate_predictions(predictions)
        if not validation_results['passed']:
            self.logger.warning(f"Validación de predicciones falló: {validation_results['errors']}")
        
        return predictions
    
    def save_model(self, model_path: str = "models/pd_model.pkl") -> None:
        """Guardar modelo entrenado."""
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        
        model_artifacts = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'config': self.config,
            'training_metrics': self.training_metrics,
            'feature_importance': self.feature_importance
        }
        
        joblib.dump(model_artifacts, model_path)
        log_operation(f"Modelo PD guardado en: {model_path}")
    
    def load_model(self, model_path: str = "models/pd_model.pkl") -> None:
        """Cargar modelo previamente entrenado."""
        model_artifacts = joblib.load(model_path)
        
        self.model = model_artifacts['model']
        self.scaler = model_artifacts['scaler']
        self.feature_columns = model_artifacts['feature_columns']
        self.training_metrics = model_artifacts.get('training_metrics', {})
        self.feature_importance = model_artifacts.get('feature_importance', {})
        
        log_operation(f"Modelo PD cargado desde: {model_path}")
    
    def generate_explainability(self, X_sample: pd.DataFrame, max_features: int = 20) -> Dict[str, Any]:
        """
        Generar explicabilidad del modelo usando SHAP.
        
        Args:
            X_sample: Muestra de datos para explicabilidad
            max_features: Máximo número de features a mostrar
            
        Returns:
            Diccionario con valores SHAP y análisis
        """
        if not self.config.get('explainability', {}).get('enable_shap', False):
            return {}
        
        try:
            # Preparar datos
            X_scaled = self.scaler.transform(X_sample[self.feature_columns])
            
            # Crear explainer
            explainer = shap.TreeExplainer(self.model)
            shap_values = explainer.shap_values(X_scaled)
            
            # Si es clasificación binaria, tomar valores para clase positiva
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
            
            # Feature importance global
            feature_importance_shap = np.abs(shap_values).mean(axis=0)
            top_features_idx = np.argsort(feature_importance_shap)[-max_features:]
            
            explainability_results = {
                'global_feature_importance': dict(zip(
                    [self.feature_columns[i] for i in top_features_idx],
                    feature_importance_shap[top_features_idx]
                )),
                'shap_values_sample': shap_values[:100].tolist(),  # Primeras 100 observaciones
                'base_value': explainer.expected_value,
                'feature_names': self.feature_columns
            }
            
            return explainability_results
            
        except Exception as e:
            self.logger.warning(f"Error generando explicabilidad: {e}")
            return {}
    
    def generate_report(self) -> str:
        """Generar reporte de entrenamiento del modelo PD."""
        if not self.training_metrics:
            return "No hay métricas de entrenamiento disponibles."
        
        report_content = f"""# Reporte Modelo PD (Probability of Default)

## Información General
- **Fecha de entrenamiento**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **Algoritmo**: {self.config.get('model', {}).get('algorithm', 'N/A')}
- **Features utilizadas**: {len(self.feature_columns)}

## Métricas de Performance

### Conjunto de Entrenamiento
- **AUC-ROC**: {self.training_metrics.get('train', {}).get('roc_auc', 0):.4f}
- **Precision**: {self.training_metrics.get('train', {}).get('precision', 0):.4f}
- **Recall**: {self.training_metrics.get('train', {}).get('recall', 0):.4f}
- **F1-Score**: {self.training_metrics.get('train', {}).get('f1_score', 0):.4f}
- **Gini**: {self.training_metrics.get('train', {}).get('gini_coefficient', 0):.4f}
- **KS**: {self.training_metrics.get('train', {}).get('ks_statistic', 0):.4f}

### Conjunto de Validación
- **AUC-ROC**: {self.training_metrics.get('validation', {}).get('roc_auc', 0):.4f}
- **Precision**: {self.training_metrics.get('validation', {}).get('precision', 0):.4f}
- **Recall**: {self.training_metrics.get('validation', {}).get('recall', 0):.4f}
- **F1-Score**: {self.training_metrics.get('validation', {}).get('f1_score', 0):.4f}

### Conjunto de Prueba
- **AUC-ROC**: {self.training_metrics.get('test', {}).get('roc_auc', 0):.4f}
- **Precision**: {self.training_metrics.get('test', {}).get('precision', 0):.4f}
- **Recall**: {self.training_metrics.get('test', {}).get('recall', 0):.4f}
- **F1-Score**: {self.training_metrics.get('test', {}).get('f1_score', 0):.4f}

## Validación Cruzada
"""
        
        if 'cross_validation' in self.training_metrics:
            cv_metrics = self.training_metrics['cross_validation']
            report_content += f"""
- **AUC promedio**: {cv_metrics.get('mean_cv_auc', 0):.4f} ± {cv_metrics.get('std_cv_auc', 0):.4f}
"""
        
        # Top features
        if self.feature_importance:
            top_features = sorted(self.feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]
            report_content += f"""
## Top 10 Features más Importantes

"""
            for i, (feature, importance) in enumerate(top_features, 1):
                report_content += f"{i}. **{feature}**: {importance:.4f}\n"
        
        return report_content