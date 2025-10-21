# Binary Bot - Mejoras v2.0

## 🚀 Mejoras Implementadas

Este documento describe las mejoras significativas realizadas al sistema binary-bot.

### 1. ✅ Gestión Segura de Credenciales

**Antes:**
```python
USERNAME = "hardcoded_email"  # ❌ Inseguro
PASSWORD = "hardcoded_pass"   # ❌ Expuesto en código
```

**Después:**
```python
IQ_USERNAME = os.getenv('IQ_USERNAME')  # ✅ Variables de entorno
IQ_PASSWORD = os.getenv('IQ_PASSWORD')  # ✅ Seguro
```

**Uso:**
```bash
export IQ_USERNAME="tu_email@example.com"
export IQ_PASSWORD="tu_contraseña"
export IQ_BALANCE_MODE="PRACTICE"  # o "REAL"
```

---

### 2. 📝 Sistema de Logging Profesional

**Características:**
- Logs estructurados con timestamps
- Niveles de log (DEBUG, INFO, WARNING, ERROR)
- Trazas de errores completas
- Monitoreo de operaciones en tiempo real

**Ejemplo:**
```
2025-10-21 10:30:15 - iq - INFO - Successfully connected to IQ Option
2025-10-21 10:30:16 - iq - INFO - Balance mode set to: PRACTICE
2025-10-21 10:30:45 - training - INFO - Training complete! Final validation accuracy: 0.6234
```

---

### 3. 🛡️ Límites de Seguridad en Trading

**Nuevas protecciones:**

| Protección | Valor por Defecto | Descripción |
|-----------|-------------------|-------------|
| `MAX_BET_AMOUNT` | $100 | Límite máximo de apuesta |
| `MAX_MARTINGALE_STEPS` | 5 | Máximo de pérdidas consecutivas |
| `MIN_CONFIDENCE` | 0.5 | Confianza mínima para operar |

**Código:**
```python
# Límite de apuesta
if bet_money > MAX_BET_AMOUNT:
    logger.warning(f"Bet exceeds max, resetting")
    bet_money = initial_bet

# Límite de Martingale
if consecutive_losses >= MAX_MARTINGALE_STEPS:
    logger.warning(f"Max losses reached, resetting")
    bet_money = initial_bet
    consecutive_losses = 0
```

---

### 4. 🔧 Código Deprecado Actualizado

**Pandas:**
```python
# Antes (deprecado)
df.fillna(method="ffill", inplace=True)
df = df.append(new_data)

# Después (moderno)
df = df.ffill()
df = pd.concat([df, new_data], ignore_index=False)
```

**TensorFlow/Keras:**
```python
# Antes (deprecado)
optimizer = Adam(lr=0.001, decay=5e-5)
checkpoint = ModelCheckpoint(monitor='val_acc')

# Después (moderno)
optimizer = Adam(learning_rate=0.001, weight_decay=5e-5)
checkpoint = ModelCheckpoint(monitor='val_accuracy')
```

---

### 5. ⚙️ Archivo de Configuración Centralizado

**Nuevo archivo:** `config.py`

```python
# Parámetros del modelo
SEQ_LEN = 5
FUTURE_PERIOD_PREDICT = 2
LEARNING_RATE = 0.001
EPOCHS = 40
BATCH_SIZE = 16

# Límites de seguridad
MAX_BET_AMOUNT = 100.0
MAX_MARTINGALE_STEPS = 5

# Indicadores técnicos
MA_WINDOWS = [20, 50]
RSI_PERIOD = 14
```

**Beneficios:**
- Configuración centralizada
- Fácil de modificar
- Validación de parámetros
- Advertencias automáticas

---

### 6. 🎯 Manejo Robusto de Errores

**Características:**

**Validación de datos:**
```python
if df is None or df.empty:
    logger.error("No data retrieved")
    return None
```

**Manejo de excepciones:**
```python
try:
    balance = iq.get_balance()
    logger.info(f"Balance: ${balance}")
except Exception as e:
    logger.error(f"Error getting balance: {e}", exc_info=True)
    return None
```

**Reintentos automáticos:**
- Reconexión en caso de fallo
- Validación de respuestas de API
- Fallback a valores seguros

---

### 7. 📊 Mejoras en el Modelo LSTM

**Arquitectura mejorada:**
```python
model = Sequential([
    LSTM(128, return_sequences=True, input_shape=...),
    Dropout(0.2),
    BatchNormalization(),

    LSTM(128, return_sequences=True),
    Dropout(0.1),
    BatchNormalization(),

    LSTM(128),
    Dropout(0.2),
    BatchNormalization(),

    Dense(32, activation='relu'),
    Dropout(0.2),

    Dense(2, activation='softmax')
])
```

**Nuevas características:**
- Early stopping (detiene entrenamiento si no mejora)
- Mejor balance de clases
- Validación temporal (no aleatoria)
- Métricas de rendimiento mejoradas

---

### 8. 📈 Indicadores Técnicos

**Indicadores implementados:**

| Indicador | Descripción |
|-----------|-------------|
| MA (20, 50) | Medias móviles |
| EMA (20, 50) | Medias móviles exponenciales |
| RSI (14) | Índice de fuerza relativa |
| Stochastic | Oscilador estocástico |

**Código reutilizable:**
```python
def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula todos los indicadores técnicos"""
    df['MA_20'] = df['close'].rolling(window=20).mean()
    df['MA_50'] = df['close'].rolling(window=50).mean()
    df['EMA_20'] = df['close'].ewm(span=20, adjust=False).mean()
    df['EMA_50'] = df['close'].ewm(span=50, adjust=False).mean()
    # ... RSI, Stochastic, etc.
    return df
```

---

## 📖 Uso del Sistema Mejorado

### Instalación

```bash
# 1. Instalar dependencias
pip install -U git+git://github.com/Lu-Yi-Hsun/iqoptionapi.git
pip install tensorflow pandas numpy scikit-learn

# 2. Configurar credenciales
cp .env.example .env
nano .env  # Editar con tus credenciales

# 3. Exportar variables (o usar archivo .env)
export IQ_USERNAME="tu_email@example.com"
export IQ_PASSWORD="tu_contraseña"
export IQ_BALANCE_MODE="PRACTICE"
```

### Entrenamiento

```bash
python training.py
```

**Salida:**
```
2025-10-21 10:30:00 - training - INFO - Starting training process
2025-10-21 10:30:15 - training - INFO - Data shape: (1000, 15)
2025-10-21 10:30:20 - training - INFO - Calculating technical indicators...
2025-10-21 10:35:00 - training - INFO - Train data: 450 sequences
2025-10-21 10:40:00 - training - INFO - Training complete! Accuracy: 0.6234
```

### Trading Automatizado

```bash
# Modo práctica con valores por defecto
python testing.py

# Modo personalizado
python testing.py EURUSD 1 2
#                 ^      ^ ^
#                 |      | |
#                 Par    | Martingale (2x)
#                        |
#                        Apuesta inicial ($1)
```

**Salida:**
```
2025-10-21 11:00:00 - testing - WARNING - BINARY BOT - AUTOMATED TRADING
2025-10-21 11:00:01 - testing - INFO - Trading: EURUSD, Bet: $1, Martingale: 2x
2025-10-21 11:00:30 - testing - INFO - PUT: 0.6234, CALL: 0.3766
2025-10-21 11:01:00 - testing - INFO - PUT $1
2025-10-21 11:01:03 - testing - INFO - Result: win
2025-10-21 11:01:03 - testing - INFO - Balance: $10001.85
```

---

## 🔒 Consideraciones de Seguridad

### ⚠️ ADVERTENCIAS CRÍTICAS

1. **Nunca compartas tu archivo `.env`** con credenciales
2. **Siempre usa modo PRACTICE primero** antes de REAL
3. **No operes con más dinero del que puedas perder**
4. **Revisa los límites de seguridad** en `config.py`
5. **Monitorea constantemente el bot** - no lo dejes desatendido

### Límites Recomendados

```python
# Para principiantes
MAX_BET_AMOUNT = 10.0          # Máximo $10 por apuesta
MAX_MARTINGALE_STEPS = 3       # Máximo 3 pérdidas consecutivas
MIN_CONFIDENCE = 0.6           # 60% de confianza mínima

# Para usuarios avanzados (⚠️ Mayor riesgo)
MAX_BET_AMOUNT = 50.0
MAX_MARTINGALE_STEPS = 5
MIN_CONFIDENCE = 0.55
```

---

## 📊 Estructura del Proyecto

```
binary-bot/
├── config.py              # ✅ Nuevo: Configuración centralizada
├── iq.py                  # ✅ Mejorado: API con logging y errores
├── training.py            # ✅ Mejorado: Código moderno, early stopping
├── testing.py             # ✅ Mejorado: Límites de seguridad
├── .env.example           # ✅ Nuevo: Plantilla de configuración
├── README.md              # Original
├── README_IMPROVEMENTS.md # ✅ Nuevo: Este documento
├── models/                # Modelos entrenados
└── logs/                  # Logs de TensorBoard

```

---

## 🐛 Problemas Comunes y Soluciones

### Error: "Missing credentials"
```bash
# Solución: Exportar variables de entorno
export IQ_USERNAME="tu_email"
export IQ_PASSWORD="tu_pass"
```

### Error: "No module named 'config'"
```bash
# Solución: Verificar que config.py existe
ls config.py
```

### Error: "Bet amount exceeds maximum"
```bash
# Solución: Ajustar en config.py
MAX_BET_AMOUNT = 100.0  # Aumentar límite
```

### Modelo no mejora (accuracy baja)
```python
# Soluciones:
# 1. Aumentar datos de entrenamiento
# 2. Ajustar hiperparámetros en config.py
EPOCHS = 60  # Más epochs
BATCH_SIZE = 32  # Batch más grande

# 3. Probar diferentes pares de divisas
ACTIVE_PAIRS = ['EURUSD', 'GBPUSD']  # Reducir pares
```

---

## 📝 Changelog

### v2.0 (2025-10-21)
- ✅ Gestión segura de credenciales con variables de entorno
- ✅ Sistema de logging profesional
- ✅ Límites de seguridad en trading
- ✅ Actualización de código deprecado (Pandas, TensorFlow)
- ✅ Archivo de configuración centralizado
- ✅ Manejo robusto de errores
- ✅ Mejoras en arquitectura LSTM
- ✅ Early stopping y mejor validación
- ✅ Documentación completa

### v1.0 (Original)
- Bot básico de trading
- Modelo LSTM simple
- Sin gestión de credenciales
- Sin límites de seguridad

---

## 🤝 Contribuciones

Las mejoras incluyen:
- Seguridad mejorada
- Código más mantenible
- Mejor rendimiento del modelo
- Protección contra pérdidas
- Documentación completa

---

## 📜 Licencia

MIT License - Mismo que el proyecto original

---

## ⚠️ Disclaimer

**ESTE SOFTWARE ES SOLO PARA FINES EDUCATIVOS**

- El trading de opciones binarias conlleva riesgos financieros
- No hay garantía de ganancias
- Puedes perder todo tu capital
- La estrategia Martingale puede acelerar las pérdidas
- Siempre usa modo PRACTICE primero
- Solo opera con dinero que puedas permitirte perder

**Los autores no se responsabilizan por pérdidas financieras.**
