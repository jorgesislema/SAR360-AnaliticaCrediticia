-- ============================================================================
-- ESQUEMA DIMENSIONAL PARA ANALÍTICA DE RIESGO CREDITICIO
-- Modelo: Estrella Optimizado para Dashboard y Análisis SAR
-- Fecha: 15 de octubre de 2025
-- ============================================================================

-- Tabla de Dimensión Tiempo
-- Granularidad: Día con jerarquías para análisis temporal
CREATE TABLE IF NOT EXISTS DIM_TIEMPO (
  id_fecha INT PRIMARY KEY,
  fecha DATE NOT NULL,
  anio INT NOT NULL,
  mes INT NOT NULL,
  trimestre INT NOT NULL,
  semana INT NOT NULL,
  dia_semana INT NOT NULL,
  mes_nombre VARCHAR(15) NOT NULL,
  trimestre_nombre VARCHAR(10) NOT NULL,
  es_fin_mes BOOLEAN DEFAULT FALSE,
  es_dia_habil BOOLEAN DEFAULT TRUE,
  periodo_regulatorio VARCHAR(10), -- Para reportes SAR
  fecha_creacion TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  INDEX idx_dim_tiempo_anio_mes (anio, mes),
  INDEX idx_dim_tiempo_trimestre (anio, trimestre)
);

-- Tabla de Dimensión Cliente
-- Incluye datos demográficos y segmentación de riesgo (SCD Tipo 2)
CREATE TABLE IF NOT EXISTS DIM_CLIENTE (
  id_cliente INT PRIMARY KEY AUTO_INCREMENT,
  codigo_cliente VARCHAR(50) UNIQUE NOT NULL,
  edad INT NOT NULL,
  grupo_edad VARCHAR(20) NOT NULL, -- 18-25, 26-35, 36-50, 51-65, 65+
  ingreso_anual DECIMAL(12,2) NOT NULL,
  bracket_ingreso VARCHAR(20) NOT NULL, -- Low, Medium, High
  tipo_vivienda VARCHAR(30) NOT NULL, -- RENT, OWN, MORTGAGE, OTHER
  anos_empleo INT NOT NULL,
  categoria_empleo VARCHAR(30), -- Nuevo, Junior, Senior, Experto
  longitud_historial_crediticio INT NOT NULL,
  tiene_default_previo BOOLEAN DEFAULT FALSE,
  segmento_cliente VARCHAR(30) NOT NULL, -- Segmentación de negocio
  score_crediticio DECIMAL(5,2), -- Score interno del cliente
  region VARCHAR(50),
  estado VARCHAR(50),
  ciudad VARCHAR(50),
  -- Campos SCD Tipo 2
  fecha_inicio DATE NOT NULL,
  fecha_fin DATE DEFAULT '9999-12-31',
  es_actual BOOLEAN DEFAULT TRUE,
  version INT DEFAULT 1,
  fecha_creacion TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  fecha_actualizacion TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  
  INDEX idx_dim_cliente_codigo (codigo_cliente),
  INDEX idx_dim_cliente_bracket (bracket_ingreso),
  INDEX idx_dim_cliente_edad (grupo_edad),
  INDEX idx_dim_cliente_actual (es_actual),
  INDEX idx_dim_cliente_segmento (segmento_cliente)
);

-- Tabla de Dimensión Producto de Crédito
-- Información del producto crediticio y sus características
CREATE TABLE IF NOT EXISTS DIM_PRODUCTO_CREDITO (
  id_producto INT PRIMARY KEY AUTO_INCREMENT,
  codigo_producto VARCHAR(30) UNIQUE NOT NULL,
  linea_negocio VARCHAR(50) NOT NULL,
  tipo_producto VARCHAR(50) NOT NULL,
  proposito_credito VARCHAR(50) NOT NULL, -- PERSONAL, EDUCATION, MEDICAL, etc.
  grado_credito CHAR(1) NOT NULL, -- A, B, C, D, E, F, G
  categoria_riesgo VARCHAR(30) NOT NULL, -- Bajo, Medio, Alto, Extremo
  tasa_interes_base DECIMAL(5,2),
  plazo_maximo_meses INT,
  monto_minimo DECIMAL(12,2),
  monto_maximo DECIMAL(12,2),
  requiere_colateral BOOLEAN DEFAULT FALSE,
  tipo_colateral VARCHAR(50),
  descripcion TEXT,
  activo BOOLEAN DEFAULT TRUE,
  fecha_creacion TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  fecha_actualizacion TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  
  INDEX idx_dim_producto_grado (grado_credito),
  INDEX idx_dim_producto_proposito (proposito_credito),
  INDEX idx_dim_producto_linea (linea_negocio),
  INDEX idx_dim_producto_categoria (categoria_riesgo)
);

-- Tabla de Dimensión Riesgo
-- Métricas y clasificaciones de riesgo para modelos PD/LGD/EAD
CREATE TABLE IF NOT EXISTS DIM_RIESGO (
  id_riesgo INT PRIMARY KEY AUTO_INCREMENT,
  codigo_riesgo VARCHAR(30) UNIQUE NOT NULL,
  rating_maestro VARCHAR(10) NOT NULL, -- AAA, AA, A, BBB, etc.
  grado_interno CHAR(1) NOT NULL, -- A, B, C, D, E, F, G
  sub_grado VARCHAR(5), -- A1, A2, B1, B2, etc.
  banda_pd VARCHAR(20) NOT NULL, -- 0-1%, 1-3%, 3-5%, etc.
  pd_minimo DECIMAL(6,4) NOT NULL,
  pd_maximo DECIMAL(6,4) NOT NULL,
  banda_lgd VARCHAR(20) NOT NULL, -- 0-20%, 20-40%, etc.
  lgd_esperada DECIMAL(5,4) NOT NULL,
  categoria_regulatoria VARCHAR(10) NOT NULL, -- A, B, C, D, E (SAR)
  requiere_provision_especial BOOLEAN DEFAULT FALSE,
  activo BOOLEAN DEFAULT TRUE,
  fecha_creacion TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  
  INDEX idx_dim_riesgo_rating (rating_maestro),
  INDEX idx_dim_riesgo_grado (grado_interno),
  INDEX idx_dim_riesgo_categoria (categoria_regulatoria),
  INDEX idx_dim_riesgo_banda_pd (banda_pd)
);

-- Tabla de Hechos Principal: Préstamos y su evolución
-- Granularidad: Un registro por préstamo por mes
CREATE TABLE IF NOT EXISTS FACT_PRESTAMOS (
  id_prestamo BIGINT PRIMARY KEY AUTO_INCREMENT,
  numero_prestamo VARCHAR(50) UNIQUE NOT NULL,
  id_fecha INT NOT NULL,
  id_cliente INT NOT NULL,
  id_producto INT NOT NULL,
  id_riesgo INT NOT NULL,
  id_fecha_originacion INT NOT NULL,
  
  -- Métricas del préstamo
  monto_original DECIMAL(12,2) NOT NULL,
  monto_actual DECIMAL(12,2) NOT NULL,
  saldo_pendiente DECIMAL(12,2) NOT NULL,
  tasa_interes DECIMAL(5,2) NOT NULL,
  plazo_meses INT NOT NULL,
  plazo_restante INT NOT NULL,
  
  -- Estado del préstamo
  estado_prestamo VARCHAR(30) NOT NULL, -- VIGENTE, VENCIDO, DEFAULT, PAGADO
  dias_mora INT DEFAULT 0,
  categoria_mora VARCHAR(20), -- AL_DIA, MORA_30, MORA_60, MORA_90, DEFAULT
  
  -- Métricas de riesgo (PD/LGD/EAD)
  probabilidad_default DECIMAL(6,4), -- PD calculada
  perdida_dado_default DECIMAL(5,4), -- LGD estimada
  exposicion_en_default DECIMAL(12,2), -- EAD calculada
  perdida_esperada DECIMAL(12,2), -- EL = PD × LGD × EAD
  score_riesgo DECIMAL(6,2), -- Score compuesto
  
  -- Métricas financieras
  ingreso_cliente DECIMAL(12,2),
  ratio_deuda_ingreso DECIMAL(5,4),
  ratio_prestamo_ingreso DECIMAL(5,4),
  pago_mensual_estimado DECIMAL(10,2),
  
  -- Indicadores de performance
  en_default BOOLEAN DEFAULT FALSE,
  provision_requerida DECIMAL(12,2) DEFAULT 0,
  provision_constituida DECIMAL(12,2) DEFAULT 0,
  
  -- Métricas de cobranza
  total_pagos_realizados DECIMAL(12,2) DEFAULT 0,
  total_intereses_pagados DECIMAL(12,2) DEFAULT 0,
  total_capital_pagado DECIMAL(12,2) DEFAULT 0,
  
  -- Campos de auditoría
  fecha_proceso DATE NOT NULL,
  usuario_proceso VARCHAR(50),
  fecha_creacion TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  fecha_actualizacion TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  
  -- Claves foráneas
  FOREIGN KEY (id_fecha) REFERENCES DIM_TIEMPO(id_fecha),
  FOREIGN KEY (id_cliente) REFERENCES DIM_CLIENTE(id_cliente),
  FOREIGN KEY (id_producto) REFERENCES DIM_PRODUCTO_CREDITO(id_producto),
  FOREIGN KEY (id_riesgo) REFERENCES DIM_RIESGO(id_riesgo),
  FOREIGN KEY (id_fecha_originacion) REFERENCES DIM_TIEMPO(id_fecha),
  
  -- Índices para performance
  INDEX idx_fact_prestamos_fecha (id_fecha),
  INDEX idx_fact_prestamos_cliente (id_cliente),
  INDEX idx_fact_prestamos_producto (id_producto),
  INDEX idx_fact_prestamos_riesgo (id_riesgo),
  INDEX idx_fact_prestamos_estado (estado_prestamo),
  INDEX idx_fact_prestamos_default (en_default),
  INDEX idx_fact_prestamos_mora (categoria_mora),
  INDEX idx_fact_prestamos_numero (numero_prestamo),
  INDEX idx_fact_prestamos_originacion (id_fecha_originacion),
  
  -- Índices compuestos para consultas frecuentes
  INDEX idx_fact_fecha_estado (id_fecha, estado_prestamo),
  INDEX idx_fact_cliente_fecha (id_cliente, id_fecha),
  INDEX idx_fact_producto_fecha (id_producto, id_fecha),
  INDEX idx_fact_riesgo_fecha (id_riesgo, id_fecha)
);

-- Tabla de Hechos: Pagos realizados
-- Granularidad: Un registro por pago
CREATE TABLE IF NOT EXISTS FACT_PAGOS (
  id_pago BIGINT PRIMARY KEY AUTO_INCREMENT,
  numero_prestamo VARCHAR(50) NOT NULL,
  id_fecha_pago INT NOT NULL,
  numero_cuota INT NOT NULL,
  
  -- Montos del pago
  monto_total_pago DECIMAL(10,2) NOT NULL,
  monto_capital DECIMAL(10,2) NOT NULL,
  monto_interes DECIMAL(10,2) NOT NULL,
  monto_mora DECIMAL(10,2) DEFAULT 0,
  monto_otros_cargos DECIMAL(10,2) DEFAULT 0,
  
  -- Estado del pago
  estado_pago VARCHAR(20) NOT NULL, -- APLICADO, REVERSADO, PENDIENTE
  dias_retraso INT DEFAULT 0,
  canal_pago VARCHAR(30), -- BANCO, ONLINE, EFECTIVO, etc.
  
  fecha_creacion TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  
  FOREIGN KEY (id_fecha_pago) REFERENCES DIM_TIEMPO(id_fecha),
  
  INDEX idx_fact_pagos_prestamo (numero_prestamo),
  INDEX idx_fact_pagos_fecha (id_fecha_pago),
  INDEX idx_fact_pagos_estado (estado_pago)
);

-- Tabla de Hechos: Provisiones mensuales
-- Granularidad: Un registro por préstamo por mes
CREATE TABLE IF NOT EXISTS FACT_PROVISIONES (
  id_provision BIGINT PRIMARY KEY AUTO_INCREMENT,
  numero_prestamo VARCHAR(50) NOT NULL,
  id_fecha INT NOT NULL,
  
  -- Provisiones calculadas
  provision_anterior DECIMAL(12,2) DEFAULT 0,
  provision_calculada DECIMAL(12,2) NOT NULL,
  provision_constituida DECIMAL(12,2) NOT NULL,
  ajuste_provision DECIMAL(12,2) DEFAULT 0,
  
  -- Motivo del ajuste
  motivo_ajuste VARCHAR(100),
  tipo_provision VARCHAR(30), -- INDIVIDUAL, GENERAL, ESPECIFICA
  metodologia VARCHAR(50), -- MODELO_INTERNO, REGULATORIO, MANUAL
  
  fecha_creacion TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  
  FOREIGN KEY (id_fecha) REFERENCES DIM_TIEMPO(id_fecha),
  
  INDEX idx_fact_provisiones_prestamo (numero_prestamo),
  INDEX idx_fact_provisiones_fecha (id_fecha),
  INDEX idx_fact_provisiones_tipo (tipo_provision)
);

-- ============================================================================
-- DATOS MAESTROS INICIALES
-- ============================================================================

-- Insertar datos iniciales en DIM_TIEMPO (últimos 3 años + próximos 2 años)
INSERT INTO DIM_TIEMPO (id_fecha, fecha, anio, mes, trimestre, semana, dia_semana, mes_nombre, trimestre_nombre, es_fin_mes, es_dia_habil, periodo_regulatorio)
SELECT 
  YEAR(fecha) * 10000 + MONTH(fecha) * 100 + DAY(fecha) as id_fecha,
  fecha,
  YEAR(fecha) as anio,
  MONTH(fecha) as mes,
  QUARTER(fecha) as trimestre,
  WEEK(fecha) as semana,
  DAYOFWEEK(fecha) as dia_semana,
  MONTHNAME(fecha) as mes_nombre,
  CONCAT('Q', QUARTER(fecha)) as trimestre_nombre,
  CASE WHEN DAY(LAST_DAY(fecha)) = DAY(fecha) THEN TRUE ELSE FALSE END as es_fin_mes,
  CASE WHEN DAYOFWEEK(fecha) IN (1, 7) THEN FALSE ELSE TRUE END as es_dia_habil,
  CASE WHEN MONTH(fecha) IN (3,6,9,12) AND DAY(LAST_DAY(fecha)) = DAY(fecha) THEN 'TRIMESTRE' ELSE 'MENSUAL' END as periodo_regulatorio
FROM (
  SELECT DATE_ADD('2023-01-01', INTERVAL seq.seq DAY) as fecha
  FROM (
    SELECT a.N + b.N * 10 + c.N * 100 + d.N * 1000 as seq
    FROM (SELECT 0 as N UNION ALL SELECT 1 UNION ALL SELECT 2 UNION ALL SELECT 3 UNION ALL SELECT 4 UNION ALL SELECT 5 UNION ALL SELECT 6 UNION ALL SELECT 7 UNION ALL SELECT 8 UNION ALL SELECT 9) a
    CROSS JOIN (SELECT 0 as N UNION ALL SELECT 1 UNION ALL SELECT 2 UNION ALL SELECT 3 UNION ALL SELECT 4 UNION ALL SELECT 5 UNION ALL SELECT 6 UNION ALL SELECT 7 UNION ALL SELECT 8 UNION ALL SELECT 9) b
    CROSS JOIN (SELECT 0 as N UNION ALL SELECT 1 UNION ALL SELECT 2 UNION ALL SELECT 3 UNION ALL SELECT 4 UNION ALL SELECT 5 UNION ALL SELECT 6 UNION ALL SELECT 7 UNION ALL SELECT 8 UNION ALL SELECT 9) c
    CROSS JOIN (SELECT 0 as N UNION ALL SELECT 1 UNION ALL SELECT 2 UNION ALL SELECT 3 UNION ALL SELECT 4) d
  ) seq
  WHERE DATE_ADD('2023-01-01', INTERVAL seq.seq DAY) <= '2027-12-31'
) fechas
ON DUPLICATE KEY UPDATE fecha = VALUES(fecha);

-- Insertar datos de referencia en DIM_RIESGO
INSERT INTO DIM_RIESGO (codigo_riesgo, rating_maestro, grado_interno, sub_grado, banda_pd, pd_minimo, pd_maximo, banda_lgd, lgd_esperada, categoria_regulatoria, requiere_provision_especial) VALUES
('RISK_A1', 'AAA', 'A', 'A1', '0-1%', 0.0000, 0.0100, '0-20%', 0.1000, 'A', FALSE),
('RISK_A2', 'AA', 'A', 'A2', '1-2%', 0.0100, 0.0200, '0-20%', 0.1500, 'A', FALSE),
('RISK_B1', 'A', 'B', 'B1', '2-3%', 0.0200, 0.0300, '10-30%', 0.2000, 'A', FALSE),
('RISK_B2', 'BBB', 'B', 'B2', '3-5%', 0.0300, 0.0500, '15-35%', 0.2500, 'B', FALSE),
('RISK_C1', 'BB', 'C', 'C1', '5-8%', 0.0500, 0.0800, '20-40%', 0.3000, 'B', FALSE),
('RISK_C2', 'B', 'C', 'C2', '8-12%', 0.0800, 0.1200, '25-45%', 0.3500, 'C', FALSE),
('RISK_D1', 'CCC', 'D', 'D1', '12-18%', 0.1200, 0.1800, '30-50%', 0.4000, 'C', TRUE),
('RISK_D2', 'CC', 'D', 'D2', '18-25%', 0.1800, 0.2500, '35-55%', 0.4500, 'D', TRUE),
('RISK_E1', 'C', 'E', 'E1', '25-35%', 0.2500, 0.3500, '40-60%', 0.5000, 'D', TRUE),
('RISK_E2', 'D', 'E', 'E2', '35-50%', 0.3500, 0.5000, '45-65%', 0.5500, 'E', TRUE),
('RISK_F1', 'D', 'F', 'F1', '50-70%', 0.5000, 0.7000, '50-70%', 0.6000, 'E', TRUE),
('RISK_F2', 'D', 'F', 'F2', '70-85%', 0.7000, 0.8500, '55-75%', 0.6500, 'E', TRUE),
('RISK_G1', 'D', 'G', 'G1', '85-95%', 0.8500, 0.9500, '60-80%', 0.7000, 'E', TRUE),
('RISK_G2', 'D', 'G', 'G2', '95-100%', 0.9500, 1.0000, '65-85%', 0.7500, 'E', TRUE);

-- Insertar productos de crédito de referencia
INSERT INTO DIM_PRODUCTO_CREDITO (codigo_producto, linea_negocio, tipo_producto, proposito_credito, grado_credito, categoria_riesgo, tasa_interes_base, plazo_maximo_meses, monto_minimo, monto_maximo, requiere_colateral, descripcion) VALUES
('PERS_A', 'Consumo', 'Personal', 'PERSONAL', 'A', 'Bajo Riesgo', 8.50, 60, 1000, 50000, FALSE, 'Préstamo personal grado A'),
('PERS_B', 'Consumo', 'Personal', 'PERSONAL', 'B', 'Riesgo Medio', 12.50, 60, 1000, 40000, FALSE, 'Préstamo personal grado B'),
('PERS_C', 'Consumo', 'Personal', 'PERSONAL', 'C', 'Riesgo Medio', 15.50, 48, 1000, 30000, FALSE, 'Préstamo personal grado C'),
('EDUC_A', 'Educación', 'Educativo', 'EDUCATION', 'A', 'Bajo Riesgo', 7.50, 84, 2000, 100000, FALSE, 'Préstamo educativo grado A'),
('EDUC_B', 'Educación', 'Educativo', 'EDUCATION', 'B', 'Riesgo Medio', 10.50, 84, 2000, 80000, FALSE, 'Préstamo educativo grado B'),
('MED_A', 'Salud', 'Médico', 'MEDICAL', 'A', 'Bajo Riesgo', 9.00, 36, 500, 25000, FALSE, 'Préstamo médico grado A'),
('MED_B', 'Salud', 'Médico', 'MEDICAL', 'B', 'Riesgo Medio', 13.00, 36, 500, 20000, FALSE, 'Préstamo médico grado B'),
('VEH_A', 'Vehículos', 'Automotriz', 'VENTURE', 'A', 'Bajo Riesgo', 6.50, 72, 5000, 200000, TRUE, 'Préstamo vehícular grado A'),
('VEH_B', 'Vehículos', 'Automotriz', 'VENTURE', 'B', 'Riesgo Medio', 9.50, 72, 5000, 150000, TRUE, 'Préstamo vehícular grado B'),
('HOME_A', 'Vivienda', 'Mejoras Hogar', 'HOMEIMPROVEMENT', 'A', 'Bajo Riesgo', 8.00, 120, 3000, 300000, TRUE, 'Préstamo mejoras hogar grado A'),
('HOME_B', 'Vivienda', 'Mejoras Hogar', 'HOMEIMPROVEMENT', 'B', 'Riesgo Medio', 11.00, 120, 3000, 200000, TRUE, 'Préstamo mejoras hogar grado B'),
('DEBT_A', 'Consolidación', 'Deuda', 'DEBTCONSOLIDATION', 'A', 'Bajo Riesgo', 9.50, 60, 2000, 80000, FALSE, 'Consolidación deuda grado A'),
('DEBT_B', 'Consolidación', 'Deuda', 'DEBTCONSOLIDATION', 'B', 'Riesgo Medio', 14.00, 60, 2000, 60000, FALSE, 'Consolidación deuda grado B');

-- ============================================================================
-- COMENTARIOS Y DOCUMENTACIÓN
-- ============================================================================

/*
ESQUEMA DIMENSIONAL - RIESGO CREDITICIO

OBJETIVO:
Modelo estrella optimizado para análisis de riesgo crediticio y dashboard gerencial
con soporte para KPIs regulatorios (SAR) y métricas de negocio.

TABLAS PRINCIPALES:
- FACT_PRESTAMOS: Tabla central con métricas de préstamos por mes
- DIM_CLIENTE: Información demográfica y segmentación (SCD Tipo 2)
- DIM_PRODUCTO_CREDITO: Catálogo de productos crediticios
- DIM_RIESGO: Clasificaciones de riesgo y bandas PD/LGD
- DIM_TIEMPO: Dimensión temporal con jerarquías

TABLAS SECUNDARIAS:
- FACT_PAGOS: Detalle de pagos realizados
- FACT_PROVISIONES: Evolución mensual de provisiones

CARACTERÍSTICAS:
- Modelo estrella para simplicidad en consultas BI
- Índices optimizados para consultas frecuentes del dashboard
- SCD Tipo 2 en clientes para historizar cambios de segmentación
- Soporte para métricas PD/LGD/EAD requeridas por Basilea III
- Compatible con regulaciones SAR de Colombia

USO RECOMENDADO:
- Dashboard ejecutivo con KPIs de riesgo
- Reportes regulatorios mensuales
- Análisis de vintage y cohortes
- Modelos predictivos PD/LGD/EAD
- Stress testing y escenarios de riesgo
*/