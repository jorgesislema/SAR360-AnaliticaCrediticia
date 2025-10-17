-- ============================================================================
-- VISTAS ANALÍTICAS PARA DASHBOARD DE RIESGO CREDITICIO
-- SAR360 - Analítica de Riesgo Crediticio
-- Fecha: 15 de octubre de 2025
-- ============================================================================

-- Vista principal: KPIs ejecutivos de riesgo crediticio
CREATE OR REPLACE VIEW VW_KPIS_EJECUTIVOS AS
SELECT 
  t.anio,
  t.mes,
  t.trimestre,
  t.mes_nombre,
  t.trimestre_nombre,
  
  -- Métricas de portfolio
  COUNT(DISTINCT f.numero_prestamo) AS total_prestamos,
  SUM(f.monto_original) AS monto_originado_total,
  SUM(f.saldo_pendiente) AS saldo_pendiente_total,
  SUM(f.exposicion_en_default) AS exposicion_total,
  
  -- Métricas de default y riesgo
  COUNT(CASE WHEN f.en_default = TRUE THEN 1 END) AS prestamos_default,
  ROUND(COUNT(CASE WHEN f.en_default = TRUE THEN 1 END) * 100.0 / COUNT(*), 2) AS tasa_default_pct,
  
  -- Portfolio at Risk (PAR)
  SUM(CASE WHEN f.dias_mora > 30 THEN f.saldo_pendiente ELSE 0 END) AS par_30_monto,
  ROUND(SUM(CASE WHEN f.dias_mora > 30 THEN f.saldo_pendiente ELSE 0 END) * 100.0 / 
        NULLIF(SUM(f.saldo_pendiente), 0), 2) AS par_30_pct,
  
  SUM(CASE WHEN f.dias_mora > 90 THEN f.saldo_pendiente ELSE 0 END) AS par_90_monto,
  ROUND(SUM(CASE WHEN f.dias_mora > 90 THEN f.saldo_pendiente ELSE 0 END) * 100.0 / 
        NULLIF(SUM(f.saldo_pendiente), 0), 2) AS par_90_pct,
  
  -- Pérdida esperada y provisiones
  SUM(f.perdida_esperada) AS perdida_esperada_total,
  SUM(f.provision_constituida) AS provision_constituida_total,
  ROUND(SUM(f.provision_constituida) * 100.0 / NULLIF(SUM(f.perdida_esperada), 0), 2) AS cobertura_provision_pct,
  
  -- Métricas promedio ponderadas
  ROUND(SUM(f.probabilidad_default * f.saldo_pendiente) / NULLIF(SUM(f.saldo_pendiente), 0), 4) AS pd_promedio_ponderada,
  ROUND(SUM(f.perdida_dado_default * f.saldo_pendiente) / NULLIF(SUM(f.saldo_pendiente), 0), 4) AS lgd_promedio_ponderada,
  
  -- ROA y rentabilidad
  SUM(f.total_intereses_pagados) AS intereses_cobrados,
  ROUND(SUM(f.total_intereses_pagados) * 100.0 / NULLIF(SUM(f.monto_original), 0), 2) AS roa_pct

FROM FACT_PRESTAMOS f
JOIN DIM_TIEMPO t ON f.id_fecha = t.id_fecha
WHERE f.estado_prestamo IN ('VIGENTE', 'VENCIDO', 'DEFAULT')
GROUP BY t.anio, t.mes, t.trimestre, t.mes_nombre, t.trimestre_nombre
ORDER BY t.anio, t.mes;

-- Vista: Análisis por segmento de cliente
CREATE OR REPLACE VIEW VW_ANALISIS_SEGMENTO_CLIENTE AS
SELECT 
  c.bracket_ingreso,
  c.grupo_edad,
  c.segmento_cliente,
  c.region,
  
  COUNT(DISTINCT f.numero_prestamo) AS total_prestamos,
  SUM(f.saldo_pendiente) AS exposure_total,
  
  -- Métricas de riesgo por segmento
  COUNT(CASE WHEN f.en_default = TRUE THEN 1 END) AS prestamos_default,
  ROUND(COUNT(CASE WHEN f.en_default = TRUE THEN 1 END) * 100.0 / COUNT(*), 2) AS tasa_default_pct,
  
  SUM(f.perdida_esperada) AS perdida_esperada,
  ROUND(SUM(f.perdida_esperada) * 100.0 / NULLIF(SUM(f.saldo_pendiente), 0), 2) AS el_rate_pct,
  
  -- Distribución de grados de riesgo
  COUNT(CASE WHEN p.grado_credito = 'A' THEN 1 END) AS grado_a_count,
  COUNT(CASE WHEN p.grado_credito = 'B' THEN 1 END) AS grado_b_count,
  COUNT(CASE WHEN p.grado_credito = 'C' THEN 1 END) AS grado_c_count,
  COUNT(CASE WHEN p.grado_credito IN ('D','E','F','G') THEN 1 END) AS grado_alto_riesgo_count,
  
  -- Ticket promedio y rentabilidad
  ROUND(AVG(f.monto_original), 2) AS ticket_promedio,
  ROUND(AVG(f.ratio_deuda_ingreso), 4) AS dti_promedio,
  ROUND(SUM(f.total_intereses_pagados) * 100.0 / NULLIF(SUM(f.monto_original), 0), 2) AS roa_segmento

FROM FACT_PRESTAMOS f
JOIN DIM_CLIENTE c ON f.id_cliente = c.id_cliente AND c.es_actual = TRUE
JOIN DIM_PRODUCTO_CREDITO p ON f.id_producto = p.id_producto
WHERE f.estado_prestamo IN ('VIGENTE', 'VENCIDO', 'DEFAULT')
GROUP BY c.bracket_ingreso, c.grupo_edad, c.segmento_cliente, c.region
ORDER BY SUM(f.saldo_pendiente) DESC;

-- Vista: Performance por línea de producto
CREATE OR REPLACE VIEW VW_PERFORMANCE_PRODUCTO AS
SELECT 
  p.linea_negocio,
  p.tipo_producto,
  p.proposito_credito,
  p.grado_credito,
  p.categoria_riesgo,
  
  COUNT(DISTINCT f.numero_prestamo) AS total_prestamos,
  SUM(f.monto_original) AS monto_originado,
  SUM(f.saldo_pendiente) AS saldo_pendiente,
  
  -- Métricas de calidad crediticia
  COUNT(CASE WHEN f.en_default = TRUE THEN 1 END) AS prestamos_default,
  ROUND(COUNT(CASE WHEN f.en_default = TRUE THEN 1 END) * 100.0 / COUNT(*), 2) AS tasa_default_pct,
  
  ROUND(AVG(f.dias_mora), 1) AS dias_mora_promedio,
  
  -- Distribución por mora
  COUNT(CASE WHEN f.categoria_mora = 'AL_DIA' THEN 1 END) AS al_dia_count,
  COUNT(CASE WHEN f.categoria_mora = 'MORA_30' THEN 1 END) AS mora_30_count,
  COUNT(CASE WHEN f.categoria_mora = 'MORA_60' THEN 1 END) AS mora_60_count,
  COUNT(CASE WHEN f.categoria_mora = 'MORA_90' THEN 1 END) AS mora_90_count,
  COUNT(CASE WHEN f.categoria_mora = 'DEFAULT' THEN 1 END) AS default_count,
  
  -- Métricas de riesgo
  ROUND(AVG(f.probabilidad_default), 4) AS pd_promedio,
  ROUND(AVG(f.perdida_dado_default), 4) AS lgd_promedio,
  SUM(f.perdida_esperada) AS perdida_esperada_total,
  
  -- Rentabilidad
  SUM(f.total_intereses_pagados) AS intereses_cobrados_total,
  ROUND(SUM(f.total_intereses_pagados) * 100.0 / NULLIF(SUM(f.monto_original), 0), 2) AS roa_producto

FROM FACT_PRESTAMOS f
JOIN DIM_PRODUCTO_CREDITO p ON f.id_producto = p.id_producto
WHERE f.estado_prestamo IN ('VIGENTE', 'VENCIDO', 'DEFAULT')
GROUP BY p.linea_negocio, p.tipo_producto, p.proposito_credito, p.grado_credito, p.categoria_riesgo
ORDER BY SUM(f.saldo_pendiente) DESC;

-- Vista: Análisis vintage (cohortes)
CREATE OR REPLACE VIEW VW_VINTAGE_ANALYSIS AS
SELECT 
  to_orig.anio AS anio_originacion,
  to_orig.trimestre AS trimestre_originacion,
  CONCAT(to_orig.anio, '-Q', to_orig.trimestre) AS cohorte,
  t.anio AS anio_observacion,
  t.mes AS mes_observacion,
  
  TIMESTAMPDIFF(MONTH, to_orig.fecha, t.fecha) AS meses_vintage,
  
  COUNT(DISTINCT f.numero_prestamo) AS prestamos_cohorte,
  SUM(f.monto_original) AS monto_originado_cohorte,
  SUM(f.saldo_pendiente) AS saldo_actual_cohorte,
  
  -- Performance acumulado de la cohorte
  COUNT(CASE WHEN f.en_default = TRUE THEN 1 END) AS defaults_acumulados,
  ROUND(COUNT(CASE WHEN f.en_default = TRUE THEN 1 END) * 100.0 / COUNT(*), 2) AS tasa_default_acumulada_pct,
  
  SUM(f.total_intereses_pagados) AS intereses_cobrados_acumulados,
  SUM(f.perdida_esperada) AS perdida_esperada_acumulada,
  
  -- Métricas de maduración
  ROUND(SUM(f.saldo_pendiente) * 100.0 / NULLIF(SUM(f.monto_original), 0), 2) AS factor_utilizacion_pct,
  ROUND(SUM(f.total_capital_pagado) * 100.0 / NULLIF(SUM(f.monto_original), 0), 2) AS amortizacion_pct

FROM FACT_PRESTAMOS f
JOIN DIM_TIEMPO t ON f.id_fecha = t.id_fecha
JOIN DIM_TIEMPO to_orig ON f.id_fecha_originacion = to_orig.id_fecha
WHERE f.estado_prestamo IN ('VIGENTE', 'VENCIDO', 'DEFAULT', 'PAGADO')
  AND TIMESTAMPDIFF(MONTH, to_orig.fecha, t.fecha) >= 0
GROUP BY to_orig.anio, to_orig.trimestre, t.anio, t.mes
ORDER BY anio_originacion, trimestre_originacion, meses_vintage;

-- Vista: Métricas PD/LGD/EAD por banda de riesgo
CREATE OR REPLACE VIEW VW_METRICAS_RIESGO AS
SELECT 
  r.rating_maestro,
  r.grado_interno,
  r.banda_pd,
  r.banda_lgd,
  r.categoria_regulatoria,
  
  COUNT(DISTINCT f.numero_prestamo) AS prestamos_banda,
  SUM(f.saldo_pendiente) AS exposure_banda,
  ROUND(SUM(f.saldo_pendiente) * 100.0 / SUM(SUM(f.saldo_pendiente)) OVER(), 2) AS participacion_exposure_pct,
  
  -- Métricas observadas vs esperadas
  ROUND(AVG(f.probabilidad_default), 4) AS pd_observada,
  ROUND(r.pd_minimo, 4) AS pd_minima_banda,
  ROUND(r.pd_maximo, 4) AS pd_maxima_banda,
  
  ROUND(AVG(f.perdida_dado_default), 4) AS lgd_observada,
  ROUND(r.lgd_esperada, 4) AS lgd_esperada_banda,
  
  -- Defaults reales vs esperados
  COUNT(CASE WHEN f.en_default = TRUE THEN 1 END) AS defaults_observados,
  ROUND(SUM(f.probabilidad_default), 0) AS defaults_esperados,
  
  -- Expected Loss
  SUM(f.perdida_esperada) AS perdida_esperada_total,
  ROUND(SUM(f.perdida_esperada) * 100.0 / NULLIF(SUM(f.saldo_pendiente), 0), 2) AS el_rate_pct,
  
  -- Provisiones
  SUM(f.provision_constituida) AS provision_constituida,
  ROUND(SUM(f.provision_constituida) * 100.0 / NULLIF(SUM(f.perdida_esperada), 0), 2) AS cobertura_pct

FROM FACT_PRESTAMOS f
JOIN DIM_RIESGO r ON f.id_riesgo = r.id_riesgo
WHERE f.estado_prestamo IN ('VIGENTE', 'VENCIDO', 'DEFAULT')
GROUP BY r.rating_maestro, r.grado_interno, r.banda_pd, r.banda_lgd, r.categoria_regulatoria
ORDER BY r.grado_interno, r.rating_maestro;

-- Vista: Resumen ejecutivo mensual para dashboard
CREATE OR REPLACE VIEW VW_DASHBOARD_RESUMEN AS
SELECT 
  t.fecha,
  t.anio,
  t.mes,
  t.mes_nombre,
  
  -- KPIs principales (números absolutos)
  COUNT(DISTINCT f.numero_prestamo) AS total_prestamos,
  ROUND(SUM(f.saldo_pendiente) / 1000000, 2) AS saldo_pendiente_millones,
  ROUND(SUM(f.perdida_esperada) / 1000000, 2) AS perdida_esperada_millones,
  
  -- KPIs principales (porcentajes)
  ROUND(COUNT(CASE WHEN f.en_default = TRUE THEN 1 END) * 100.0 / COUNT(*), 2) AS tasa_default,
  ROUND(SUM(CASE WHEN f.dias_mora > 30 THEN f.saldo_pendiente ELSE 0 END) * 100.0 / 
        NULLIF(SUM(f.saldo_pendiente), 0), 2) AS par_30,
  ROUND(SUM(f.perdida_esperada) * 100.0 / NULLIF(SUM(f.saldo_pendiente), 0), 2) AS el_rate,
  ROUND(SUM(f.provision_constituida) * 100.0 / NULLIF(SUM(f.perdida_esperada), 0), 2) AS cobertura_provision,
  
  -- Segmentación top 3
  (SELECT c.bracket_ingreso 
   FROM FACT_PRESTAMOS f2 
   JOIN DIM_CLIENTE c ON f2.id_cliente = c.id_cliente AND c.es_actual = TRUE
   WHERE f2.id_fecha = f.id_fecha 
   GROUP BY c.bracket_ingreso 
   ORDER BY SUM(f2.saldo_pendiente) DESC 
   LIMIT 1) AS top_segmento_ingreso,
   
  (SELECT p.linea_negocio 
   FROM FACT_PRESTAMOS f3 
   JOIN DIM_PRODUCTO_CREDITO p ON f3.id_producto = p.id_producto
   WHERE f3.id_fecha = f.id_fecha 
   GROUP BY p.linea_negocio 
   ORDER BY SUM(f3.saldo_pendiente) DESC 
   LIMIT 1) AS top_linea_negocio,
   
  -- Alertas y flags
  CASE WHEN COUNT(CASE WHEN f.en_default = TRUE THEN 1 END) * 100.0 / COUNT(*) > 25 
       THEN 'ALTO' ELSE 'NORMAL' END AS alerta_default,
  CASE WHEN SUM(CASE WHEN f.dias_mora > 30 THEN f.saldo_pendiente ELSE 0 END) * 100.0 / 
            NULLIF(SUM(f.saldo_pendiente), 0) > 15 
       THEN 'ALTO' ELSE 'NORMAL' END AS alerta_par,
  CASE WHEN SUM(f.provision_constituida) * 100.0 / NULLIF(SUM(f.perdida_esperada), 0) < 80 
       THEN 'BAJO' ELSE 'ADECUADO' END AS alerta_cobertura

FROM FACT_PRESTAMOS f
JOIN DIM_TIEMPO t ON f.id_fecha = t.id_fecha
WHERE f.estado_prestamo IN ('VIGENTE', 'VENCIDO', 'DEFAULT')
  AND t.es_fin_mes = TRUE  -- Solo fin de mes para reportes consolidados
GROUP BY t.fecha, t.anio, t.mes, t.mes_nombre, f.id_fecha
ORDER BY t.fecha DESC;

-- Vista: Tendencias y variaciones MoM/YoY
CREATE OR REPLACE VIEW VW_TENDENCIAS_RIESGO AS
SELECT 
  t.fecha,
  t.anio,
  t.mes,
  
  -- Métricas actuales
  ROUND(COUNT(CASE WHEN f.en_default = TRUE THEN 1 END) * 100.0 / COUNT(*), 2) AS tasa_default_actual,
  ROUND(SUM(f.saldo_pendiente) / 1000000, 2) AS exposure_actual_millones,
  ROUND(SUM(f.perdida_esperada) * 100.0 / NULLIF(SUM(f.saldo_pendiente), 0), 2) AS el_rate_actual,
  
  -- Variaciones MoM (Month over Month)
  ROUND(
    (COUNT(CASE WHEN f.en_default = TRUE THEN 1 END) * 100.0 / COUNT(*)) -
    LAG(COUNT(CASE WHEN f.en_default = TRUE THEN 1 END) * 100.0 / COUNT(*)) 
    OVER (ORDER BY t.anio, t.mes), 2
  ) AS tasa_default_var_mom,
  
  ROUND(
    (SUM(f.saldo_pendiente) / 1000000) -
    LAG(SUM(f.saldo_pendiente) / 1000000) 
    OVER (ORDER BY t.anio, t.mes), 2
  ) AS exposure_var_mom_millones,
  
  -- Variaciones YoY (Year over Year) 
  ROUND(
    (COUNT(CASE WHEN f.en_default = TRUE THEN 1 END) * 100.0 / COUNT(*)) -
    LAG(COUNT(CASE WHEN f.en_default = TRUE THEN 1 END) * 100.0 / COUNT(*), 12) 
    OVER (ORDER BY t.anio, t.mes), 2
  ) AS tasa_default_var_yoy,
  
  ROUND(
    ((SUM(f.saldo_pendiente) / 1000000) - 
     LAG(SUM(f.saldo_pendiente) / 1000000, 12) OVER (ORDER BY t.anio, t.mes)) * 100.0 /
    NULLIF(LAG(SUM(f.saldo_pendiente) / 1000000, 12) OVER (ORDER BY t.anio, t.mes), 0), 2
  ) AS exposure_crecimiento_yoy_pct,
  
  -- Promedios móviles (3 meses)
  ROUND(AVG(COUNT(CASE WHEN f.en_default = TRUE THEN 1 END) * 100.0 / COUNT(*)) 
        OVER (ORDER BY t.anio, t.mes ROWS BETWEEN 2 PRECEDING AND CURRENT ROW), 2) AS tasa_default_ma3,
  
  ROUND(AVG(SUM(f.perdida_esperada) * 100.0 / NULLIF(SUM(f.saldo_pendiente), 0)) 
        OVER (ORDER BY t.anio, t.mes ROWS BETWEEN 2 PRECEDING AND CURRENT ROW), 2) AS el_rate_ma3

FROM FACT_PRESTAMOS f
JOIN DIM_TIEMPO t ON f.id_fecha = t.id_fecha
WHERE f.estado_prestamo IN ('VIGENTE', 'VENCIDO', 'DEFAULT')
  AND t.es_fin_mes = TRUE
GROUP BY t.fecha, t.anio, t.mes
ORDER BY t.anio, t.mes;

-- ============================================================================
-- COMENTARIOS Y DOCUMENTACIÓN DE VISTAS
-- ============================================================================

/*
VISTAS ANALÍTICAS - SAR360 RIESGO CREDITICIO

PROPÓSITO:
Conjunto de vistas optimizadas para alimentar dashboard ejecutivo y reportes
regulatorios de riesgo crediticio con métricas estándar de la industria.

VISTAS INCLUIDAS:

1. VW_KPIS_EJECUTIVOS
   - Métricas consolidadas mensuales para dashboard principal
   - Incluye: Default rate, PAR 30/90, Expected Loss, Cobertura, ROA
   - Granularidad: Mensual
   - Uso: Tableros ejecutivos, reportes gerenciales

2. VW_ANALISIS_SEGMENTO_CLIENTE  
   - Análisis de riesgo por demografía y segmentación de clientes
   - Incluye: Performance por edad, ingresos, región, DTI
   - Granularidad: Por segmento
   - Uso: Estrategia comercial, pricing, políticas de crédito

3. VW_PERFORMANCE_PRODUCTO
   - Performance crediticia por línea y tipo de producto
   - Incluye: Default rates, mora, PD/LGD por producto
   - Granularidad: Por producto/grado
   - Uso: Gestión de producto, pricing, límites de exposición

4. VW_VINTAGE_ANALYSIS
   - Análisis de cohortes por fecha de originación
   - Incluye: Maduración, default curves, seasoning effects
   - Granularidad: Cohorte trimestral por mes de observación
   - Uso: Modelos de provisión, gestión de riesgo, proyecciones

5. VW_METRICAS_RIESGO
   - Métricas PD/LGD/EAD por banda de rating interno
   - Incluye: Comparación observado vs esperado, backtesting
   - Granularidad: Por banda de riesgo
   - Uso: Validación de modelos, recalibración, Pilar II

6. VW_DASHBOARD_RESUMEN
   - Vista principal para dashboard en tiempo real
   - Incluye: KPIs principales, alertas, flags de atención
   - Granularidad: Fin de mes
   - Uso: Monitoreo diario, reportes flash, alertas automáticas

7. VW_TENDENCIAS_RIESGO
   - Tendencias temporales con variaciones MoM/YoY
   - Incluye: Promedios móviles, variaciones porcentuales
   - Granularidad: Mensual con cálculos de tendencia
   - Uso: Análisis de tendencias, forecasting, early warnings

PERFORMANCE:
- Todas las vistas incluyen índices implícitos por las FKs del modelo
- Vistas pre-agregadas para reducir tiempo de consulta en dashboard
- Filtros optimizados por estado de préstamo y fechas

MANTENIMIENTO:
- Actualización automática con refresh de FACT_PRESTAMOS
- Compatible con refresh incremental por fecha de proceso
- Validación de integridad referencial automática

REGULATORIO:
- Métricas alineadas con Basilea III (PD, LGD, EAD, EL)
- Compatible con reportes SAR Colombia
- Trazabilidad completa para auditorías externas
*/