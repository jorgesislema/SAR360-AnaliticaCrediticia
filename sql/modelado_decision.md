# Decisión de Modelado Dimensional - Analítica de Riesgo Crediticio

## Fecha: 15 de octubre de 2025
## Proyecto: SAR360-AnaliticaCrediticia

## Objetivos Analíticos y KPIs

### KPIs Principales
- **Tasa de Default Agregada**: % préstamos en default por período
- **Pérdida Esperada (EL)**: PD × LGD × EAD por segmento
- **Pérdida Inesperada (UL)**: Volatilidad de pérdidas
- **Rentabilidad Ajustada por Riesgo (RAROC)**: (Ingresos - EL) / Capital Económico
- **Provisiones vs Pérdidas Reales**: Accuracy del modelo de provisioning

### KPIs Regulatorios (SAR)
- **Índice de Morosidad por Categoría**: A, B, C, D, E
- **Cobertura de Provisiones**: Provisiones / Cartera en Riesgo
- **Concentración por Sector/Región**: Exposición máxima por segmento
- **Indicadores SARLAFT**: Alertas y reportes de operaciones inusuales

## Granularidad de Hechos

### Tabla Principal: FACT_LOANS
- **Granularidad**: Un registro por préstamo por mes
- **Volumen esperado**: ~500K préstamos × 60 meses = 30M registros
- **SLAs**: Actualización mensual, latencia máxima 2 días

### Tablas Complementarias
- **FACT_PAYMENTS**: Un registro por pago realizado
- **FACT_RECOVERIES**: Un registro por acción de cobranza
- **FACT_PROVISIONS**: Un registro por ajuste de provisiones mensual

## Dimensiones y Jerarquías

### DIM_TIEMPO
- **Jerarquía**: Año → Trimestre → Mes → Día
- **Atributos especiales**: Fin de mes, días hábiles, períodos regulatorios

### DIM_CLIENTE
- **Jerarquía**: Región → Estado → Ciudad
- **Atributos**: Segmento, score crediticio, antigüedad, ingresos_band
- **SCD Tipo 2**: Para cambios en segmento y score

### DIM_PRODUCTO
- **Jerarquía**: Línea de negocio → Producto → Sub-producto
- **Atributos**: Tasa, plazo, colateral, propósito

### DIM_RIESGO
- **Jerarquía**: Rating → Grade → Sub-grade
- **Atributos**: PD_band, LGD_band, EAD_band, vintage

### DIM_GEOGRAFIA
- **Jerarquía**: País → Región → Estado → Ciudad
- **Atributos**: PIB regional, tasa de desempleo, índices económicos

## SCD (Slowly Changing Dimensions)

### Tipo 1 (Sobrescribir)
- Datos demográficos básicos (dirección, teléfono)
- Correcciones de datos

### Tipo 2 (Historizar)
- **DIM_CLIENTE**: Cambios en segmento, ingresos, score crediticio
- **DIM_PRODUCTO**: Cambios en tasas, términos y condiciones
- **DIM_RIESGO**: Reclasificaciones de rating

## Cardinalidad Estimada

| Dimensión | Registros | Crecimiento Anual |
|-----------|-----------|-------------------|
| DIM_TIEMPO | 3,650 | 365 |
| DIM_CLIENTE | 100K | 15% |
| DIM_PRODUCTO | 50 | 5% |
| DIM_RIESGO | 100 | Estable |
| DIM_GEOGRAFIA | 500 | Estable |

## Elección: Estrella vs Copo de Nieve

### **DECISIÓN: MODELO ESTRELLA**

#### Justificación:
1. **Simplicidad para BI**: Queries más simples para dashboard Power BI
2. **Performance**: Menos JOINs para consultas analíticas frecuentes
3. **Comprensión del negocio**: Estructura más intuitiva para analistas de riesgo
4. **Volumen manejable**: Las dimensiones no son excesivamente grandes

#### Consideraciones del Copo de Nieve descartadas:
- La jerarquía geográfica se mantiene en una sola tabla (DIM_GEOGRAFIA)
- La jerarquía de productos se mantiene desnormalizada
- La reducción de redundancia no justifica la complejidad adicional

## Impacto en KPIs y Vistas

### Vistas Principales a Crear:

1. **VW_PORTFOLIO_PERFORMANCE**
   - Métricas agregadas por mes/trimestre/año
   - PD, LGD, EAD promedio por segmento

2. **VW_RISK_CONCENTRATION**
   - Concentración por geografía, sector, producto
   - Límites regulatorios vs exposición actual

3. **VW_VINTAGE_ANALYSIS**
   - Performance por cohorte de originación
   - Curvas de default por vintage

4. **VW_REGULATORY_REPORTING**
   - KPIs específicos para reportes SAR
   - Índices de morosidad por categoría

5. **VW_PROFITABILITY_ANALYSIS**
   - RAROC por segmento/producto
   - Ingresos vs pérdidas esperadas

## Consideraciones Técnicas

### Índices Propuestos:
- **FACT_LOANS**: id_fecha, id_cliente, id_producto, loan_status
- **Índices compuestos**: (id_fecha, loan_status), (id_cliente, id_fecha)

### Particionamiento:
- **FACT_LOANS**: Partición por año (id_fecha)
- **FACT_PAYMENTS**: Partición por trimestre

### Agregaciones Pre-calculadas:
- Métricas mensuales por segmento
- Ratios de concentración por geografía
- Provisiones acumuladas por vintage

## Aprobación Requerida

Este diseño requiere aprobación antes de proceder con:
1. Creación del esquema DDL
2. Definición de vistas analíticas
3. Implementación de ETL
4. Desarrollo de dashboard Power BI