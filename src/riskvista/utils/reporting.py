import pandas as pd
import numpy as np
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
import yaml

def setup_logging(log_dir: str = "logs", log_level: str = "INFO") -> logging.Logger:
    """
    Configurar logging para el proyecto.
    
    Args:
        log_dir: Directorio para logs
        log_level: Nivel de logging
        
    Returns:
        Logger configurado
    """
    Path(log_dir).mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = Path(log_dir) / f"riskvista_{timestamp}.log"
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)

def write_report(content: str, filename: str, report_dir: str = "reports", 
                format_json: Optional[Dict[str, Any]] = None) -> None:
    """
    Escribir reporte en formato markdown y opcionalmente JSON.
    
    Args:
        content: Contenido del reporte en markdown
        filename: Nombre del archivo (sin extensión)
        report_dir: Directorio de reportes
        format_json: Datos adicionales para exportar en JSON
    """
    Path(report_dir).mkdir(exist_ok=True)
    
    # Escribir markdown
    md_file = Path(report_dir) / f"{filename}.md"
    with open(md_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    # Escribir JSON si se proporciona
    if format_json:
        json_file = Path(report_dir) / f"{filename}.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(format_json, f, indent=2, ensure_ascii=False, default=str)

def log_operation(message: str, level: str = "INFO", 
                 extra_data: Optional[Dict[str, Any]] = None) -> None:
    """
    Registrar operación en logs.
    
    Args:
        message: Mensaje a registrar
        level: Nivel de log
        extra_data: Datos adicionales para log estructurado
    """
    logger = logging.getLogger(__name__)
    log_func = getattr(logger, level.lower())
    
    if extra_data:
        message = f"{message} | {json.dumps(extra_data, default=str)}"
    
    log_func(message)

def save_error_rows(df_errors: pd.DataFrame, filename: str, 
                   error_dir: str = "data/errors") -> None:
    """
    Guardar filas con errores en CSV.
    
    Args:
        df_errors: DataFrame con filas erróneas
        filename: Nombre del archivo de errores
        error_dir: Directorio de errores
    """
    Path(error_dir).mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    error_file = Path(error_dir) / f"{filename}_errors_{timestamp}.csv"
    
    df_errors.to_csv(error_file, index=False, encoding='utf-8')
    log_operation(f"Se guardaron {len(df_errors)} filas con errores en {error_file}")

def validate_data_quality(df: pd.DataFrame, rules: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validar calidad de datos según reglas definidas.
    
    Args:
        df: DataFrame a validar
        rules: Reglas de calidad de datos
        
    Returns:
        Diccionario con resultados de validación
    """
    results = {
        "total_rows": len(df),
        "validation_timestamp": datetime.now().isoformat(),
        "passed": True,
        "errors": [],
        "warnings": [],
        "metrics": {}
    }
    
    # Validar completeness
    if "completeness" in rules:
        for column, threshold in rules["completeness"].items():
            if column in df.columns:
                completeness = (1 - df[column].isnull().sum() / len(df)) * 100
                results["metrics"][f"{column}_completeness"] = completeness
                
                if completeness < float(threshold.replace("%", "")):
                    results["errors"].append(
                        f"Columna {column}: {completeness:.1f}% completeness < {threshold}"
                    )
                    results["passed"] = False
    
    # Validar validity
    if "validity" in rules:
        for column, rule in rules["validity"].items():
            if column in df.columns:
                if "<=" in rule and ">=" in rule:
                    # Rango numérico
                    parts = rule.split(" <= valor <= ")
                    min_val, max_val = float(parts[0]), float(parts[1])
                    invalid_rows = ((df[column] < min_val) | (df[column] > max_val)).sum()
                    
                    if invalid_rows > 0:
                        results["errors"].append(
                            f"Columna {column}: {invalid_rows} valores fuera del rango [{min_val}, {max_val}]"
                        )
                        results["passed"] = False
                
                elif rule.startswith("[") and rule.endswith("]"):
                    # Lista de valores válidos
                    valid_values = eval(rule)
                    invalid_rows = (~df[column].isin(valid_values)).sum()
                    
                    if invalid_rows > 0:
                        results["errors"].append(
                            f"Columna {column}: {invalid_rows} valores inválidos"
                        )
                        results["passed"] = False
    
    return results

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Cargar configuración desde archivo YAML.
    
    Args:
        config_path: Ruta al archivo de configuración
        
    Returns:
        Diccionario con configuración
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def calculate_risk_metrics(df: pd.DataFrame) -> Dict[str, float]:
    """
    Calcular métricas específicas de riesgo crediticio.
    
    Args:
        df: DataFrame con datos de préstamos
        
    Returns:
        Diccionario con métricas de riesgo
    """
    metrics = {}
    
    if 'default_flag' in df.columns:
        metrics['default_rate'] = df['default_flag'].mean()
        metrics['default_count'] = df['default_flag'].sum()
    
    if 'pd_score' in df.columns:
        metrics['avg_pd'] = df['pd_score'].mean()
        metrics['pd_std'] = df['pd_score'].std()
    
    if 'lgd_score' in df.columns:
        metrics['avg_lgd'] = df['lgd_score'].mean()
        metrics['lgd_std'] = df['lgd_score'].std()
    
    if 'ead_score' in df.columns:
        metrics['avg_ead'] = df['ead_score'].mean()
        metrics['ead_std'] = df['ead_score'].std()
    
    if all(col in df.columns for col in ['pd_score', 'lgd_score', 'ead_score', 'monto_aprobado']):
        df['expected_loss'] = df['pd_score'] * df['lgd_score'] * df['ead_score'] * df['monto_aprobado']
        metrics['total_expected_loss'] = df['expected_loss'].sum()
        metrics['el_rate'] = df['expected_loss'].sum() / df['monto_aprobado'].sum()
    
    return metrics

def generate_sar_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generar métricas específicas para reportes SAR.
    
    Args:
        df: DataFrame con datos de préstamos
        
    Returns:
        Diccionario con métricas SAR
    """
    sar_metrics = {
        "fecha_reporte": datetime.now().strftime("%Y-%m-%d"),
        "metricas": {}
    }
    
    # Índice de morosidad por categoría
    if 'categoria_riesgo' in df.columns and 'dias_mora' in df.columns:
        morosidad_por_categoria = df.groupby('categoria_riesgo').agg({
            'dias_mora': lambda x: (x > 30).mean() * 100
        }).round(2)
        
        sar_metrics["metricas"]["indice_morosidad"] = morosidad_por_categoria.to_dict()
    
    # Concentración por región
    if 'region' in df.columns and 'monto_aprobado' in df.columns:
        concentracion_region = (df.groupby('region')['monto_aprobado'].sum() / 
                               df['monto_aprobado'].sum() * 100).round(2)
        
        sar_metrics["metricas"]["concentracion_regional"] = concentracion_region.to_dict()
    
    # Cobertura de provisiones
    if 'provision_constituida' in df.columns and 'saldo_capital' in df.columns:
        cobertura = (df['provision_constituida'].sum() / 
                    df[df['dias_mora'] > 30]['saldo_capital'].sum() * 100)
        
        sar_metrics["metricas"]["cobertura_provisiones"] = round(cobertura, 2)
    
    return sar_metrics

def export_to_powerbi_format(df: pd.DataFrame, output_path: str) -> None:
    """
    Exportar datos en formato optimizado para Power BI.
    
    Args:
        df: DataFrame a exportar
        output_path: Ruta del archivo de salida
    """
    # Optimizaciones para Power BI
    df_optimized = df.copy()
    
    # Convertir fechas a formato apropiado
    date_columns = df_optimized.select_dtypes(include=['datetime64']).columns
    for col in date_columns:
        df_optimized[col] = df_optimized[col].dt.strftime('%Y-%m-%d')
    
    # Redondear decimales para reducir tamaño
    numeric_columns = df_optimized.select_dtypes(include=[np.number]).columns
    for col in numeric_columns:
        if df_optimized[col].dtype == 'float64':
            df_optimized[col] = df_optimized[col].round(4)
    
    # Exportar a CSV con UTF-8 BOM para Power BI
    df_optimized.to_csv(output_path, index=False, encoding='utf-8-sig')
    
    log_operation(f"Datos exportados para Power BI: {output_path} ({len(df_optimized)} filas)")

class ModelValidator:
    """Validador para modelos de riesgo crediticio."""
    
    def __init__(self, model_type: str):
        """
        Inicializar validador.
        
        Args:
            model_type: Tipo de modelo (pd, lgd, ead)
        """
        self.model_type = model_type.lower()
        self.validation_rules = self._load_validation_rules()
    
    def _load_validation_rules(self) -> Dict[str, Any]:
        """Cargar reglas de validación específicas por tipo de modelo."""
        rules = {
            "pd": {
                "score_range": [0, 1],
                "required_features": ["loan_amnt", "int_rate", "grade"],
                "target_variable": "default_flag"
            },
            "lgd": {
                "score_range": [0, 1],
                "required_features": ["loan_amnt", "recoveries", "total_rec_prncp"],
                "target_variable": "loss_rate"
            },
            "ead": {
                "score_range": [0, 1.5],
                "required_features": ["credit_limit", "current_balance", "utilization_rate"],
                "target_variable": "ead_ratio"
            }
        }
        return rules.get(self.model_type, {})
    
    def validate_predictions(self, y_pred: np.ndarray) -> Dict[str, Any]:
        """
        Validar predicciones del modelo.
        
        Args:
            y_pred: Predicciones del modelo
            
        Returns:
            Resultados de validación
        """
        results = {"passed": True, "errors": [], "warnings": []}
        
        # Validar rango de scores
        if "score_range" in self.validation_rules:
            min_val, max_val = self.validation_rules["score_range"]
            out_of_range = ((y_pred < min_val) | (y_pred > max_val)).sum()
            
            if out_of_range > 0:
                results["errors"].append(
                    f"{out_of_range} predicciones fuera del rango [{min_val}, {max_val}]"
                )
                results["passed"] = False
        
        # Validar distribución
        if len(np.unique(y_pred)) == 1:
            results["warnings"].append("Todas las predicciones son iguales")
        
        return results