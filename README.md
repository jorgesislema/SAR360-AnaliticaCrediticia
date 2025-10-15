# SAR360-AnaliticaCrediticia

Proyecto integral de ciencia de datos y analítica de riesgo crediticio basado en `Loan_default.csv`. Implementa modelos PD/LGD/EAD con XGBoost/LightGBM, métricas regulatorias SAR, explicabilidad avanzada y dashboard Power BI. Pipeline reproducible con gobernanza de datos y estructura modular empresarial.

## Flujo de Trabajo Gobernado

**ETL → EDA → Modelado Dimensional → ML → Dashboard → Reporte Final**

Cada fase requiere aprobación explícita antes de continuar. No se avanza sin validación completa de entregables.

## Estructura del Proyecto

```
SAR360-AnaliticaCrediticia/
├── data/                          # Datos (gitignored)
│   ├── raw/Loan_default.csv      # Dataset original
│   ├── interim/                   # Datos intermedios
│   ├── processed/                 # Datos limpios finales
│   └── errors/                    # Registros con errores
├── notebooks/                     # Jupyter notebooks por fase
├── src/riskvista/                 # Código fuente modular
│   ├── utils/                     # Utilidades (ingesta, EDA, reporting)
│   ├── features/                  # Ingeniería de características
│   └── models/                    # Modelos PD/LGD/EAD
├── configs/                       # Configuraciones YAML
│   ├── pd.yaml                    # Configuración modelo PD
│   ├── lgd.yaml                   # Configuración modelo LGD
│   └── ead.yaml                   # Configuración modelo EAD
├── sql/                           # Scripts de base de datos
│   ├── modelado_decision.md       # Decisiones de modelado dimensional
│   ├── schema.sql                 # DDL del esquema
│   └── views.sql                  # Vistas analíticas
├── schema/                        # Esquema de proyecto
│   └── project_schema.json        # Fuente de verdad del esquema
├── dashboards/powerbi/            # Artefactos Power BI
├── reports/                       # Reportes por fase
├── logs/                          # Logs de ejecución
├── models/                        # Modelos entrenados
├── tests/                         # Pruebas unitarias
├── approvals/                     # Archivos de aprobación
├── .github/workflows/             # CI/CD
└── requirements.txt               # Dependencias Python
```

## Modelos de Riesgo Crediticio

### 1. Modelo PD (Probability of Default)
- **Algoritmo**: XGBoost/LightGBM
- **Target**: `default_flag` (0/1)
- **Métricas**: AUC-ROC, Gini, KS, IV
- **Configuración**: `configs/pd.yaml`

### 2. Modelo LGD (Loss Given Default)
- **Algoritmo**: LightGBM (regresión)
- **Target**: `loss_rate` (0-1, solo defaults)
- **Métricas**: RMSE, MAE, R², directional accuracy
- **Configuración**: `configs/lgd.yaml`

### 3. Modelo EAD (Exposure at Default)
- **Algoritmo**: XGBoost (regresión)
- **Target**: `ead_ratio` (0-1.5)
- **Métricas**: RMSE, utilization accuracy, CCF error
- **Configuración**: `configs/ead.yaml`

## Métricas Regulatorias (SAR)

### Implementadas:
- **SARc**: Índices de morosidad por categoría de riesgo
- **SARL**: Concentración por región y sector económico
- **SARO**: Cobertura de provisiones vs cartera en riesgo
- **SARLAFT**: Indicadores de operaciones inusuales

### Reportes automáticos:
- Dashboard ejecutivo con KPIs regulatorios
- Alertas de concentración de riesgo
- Seguimiento de límites normativos

## Instalación y Configuración

### 1. Crear Entorno Virtual
```bash
# Windows PowerShell
python -m venv .venv_sar360
.venv_sar360\Scripts\Activate.ps1

# Instalar dependencias
pip install -r requirements.txt
```

### 2. Configurar Estructura de Datos
```bash
# Colocar Loan_default.csv en data/raw/
# El sistema detectará automáticamente el archivo
```

### 3. Configurar Base de Datos (Opcional)
```bash
# MySQL/PostgreSQL para modelo dimensional
# Esquema disponible en sql/schema.sql
```

## Uso del Sistema

### Fase 1: ETL
```python
# Ejecutar notebook de ETL
# notebooks/01_etl.ipynb
# Genera: reports/etl_report.md, data/processed/clean_data.csv
```

### Fase 2: EDA
```python
# Ejecutar análisis exploratorio
# notebooks/02_eda.ipynb  
# Genera: reports/eda_report.md, figuras en reports/
```

### Fase 3: Modelado Dimensional
```python
# Revisar decisiones de modelado
# sql/modelado_decision.md
# Aprobar estructura en approvals/MODELADO_APPROVED.txt
```

### Fase 4: Modelos ML
```python
from src.riskvista.models.train_pd import PDModel

# Entrenar modelo PD
pd_model = PDModel("configs/pd.yaml")
X, y = pd_model.prepare_data(df)
metrics = pd_model.train(X, y)
pd_model.save_model()
```

### Fase 5: Dashboard Power BI
```python
# Exportar datos para Power BI
# dashboards/powerbi/data_export.csv
# Implementar DAX según dashboards/powerbi/DAX.md
```

## Características Avanzadas

### Explicabilidad (SHAP)
- Interpretación global y local de modelos
- Feature importance por segmento
- Análisis de impacto marginal

### Validación y Calidad
- Great Expectations para validación de datos
- Pandera para schema validation
- MLflow para tracking de experimentos

### Monitoreo de Modelo
- Drift detection automático
- Performance monitoring
- Alertas de degradación

### Pipeline CI/CD
- Tests automatizados en GitHub Actions
- Validación de modelos en PR
- Deployment automático a staging

## Métricas de Negocio

### KPIs Principales:
- **Pérdida Esperada**: PD × LGD × EAD
- **RAROC**: (Ingresos - EL) / Capital Económico
- **Cobertura de Provisiones**: 95%+ sobre cartera en riesgo
- **Concentración Máxima**: 5% por región, 15% por producto

### Alertas Automáticas:
- Incremento >10% en tasa de default
- Concentración >límites regulatorios
- Drift >5% en distribución de features
- Performance <umbral en validación

## Compliance y Auditoría

### Documentación:
- Decisiones de modelado documentadas y aprobadas
- Versionado completo de modelos y datos
- Logs detallados de todas las operaciones
- Backup automático de artefactos críticos

### Reportes Regulatorios:
- Reportes mensuales automáticos SAR
- Stress testing trimestral
- Backtesting anual de modelos
- Documentación de cambios metodológicos

## Soporte y Mantenimiento

### Logs y Debugging:
- Logs estructurados en `logs/`
- Errores de datos en `data/errors/`
- Métricas de performance en tiempo real

### Testing:
```bash
# Ejecutar tests
pytest tests/ -v --cov=src/

# Tests específicos de modelos
pytest tests/test_pd_model.py
```

### Actualización de Modelos:
```bash
# Re-entrenamiento mensual automático
# Validación A/B de nuevas versiones
# Rollback automático si performance degrada
```

## Licencia y Contacto

Proyecto interno SAR360. Contactar al equipo de riesgo para acceso y soporte técnico.

---

**Versión**: 2.2  
**Última actualización**: 15 de octubre de 2025  
**Responsable técnico**: Equipo de Ciencia de Datos - Riesgo Crediticio
