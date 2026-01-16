# REPORT Strategia – Documentazione Operativa

Questo progetto genera **report di validazione strategia** a partire da file `SIGNAL_*.csv` contenenti eventi **IN / OUT**.

Il report calcola **metriche di performance, rischio e robustezza**, incluse versioni **outlier-filtered** per validare la stabilità della strategia eliminando eventi eccezionali.

---

## 1. Requisiti

* Python **>= 3.10**
* Librerie:

  * `pandas`
  * `numpy`

Ambiente consigliato: **virtualenv / venv**.

---

## 2. Struttura del progetto (essenziale)

```
REPORT strategia/
│
├─ scripts/
│   └─ report_strategia.py        # entry point
│
├─ metrics/
│   ├─ base.py                    # registry metriche
│   ├─ __init__.py                # apply_metrics + loader
│   ├─ gross_pnl_outlier_filtered.py
│   └─ ... (altre metriche)
│
├─ engine/
│   ├─ backtest.py
│   └─ equity_additive.py
│
├─ app_io/
│   ├─ loader.py                  # load_signal_csv
│   └─ exporter.py                # export_report_csv
│
└─ README.md
```

---

## 3. Input richiesto

### 3.1 File SIGNAL

* Nome: `SIGNAL_*.csv`
* Deve contenere:

  * colonna `close`
  * colonna eventi `HOLD` **oppure** `SIGNAL`
  * eventi testuali: `IN`, `OUT`

Esempio:

```
datetime,close,HOLD
2025-01-01 09:00,100,IN
2025-01-01 09:15,101,
2025-01-01 10:00,102,OUT
```

---

## 4. Modalità di lancio

### 4.1 Avvio standard (interattivo – consigliato)

```bash
python scripts/report_strategia.py
```

Il programma chiederà:

1. **Equity Start** (default 100)
2. Conferma cartella file SIGNAL
3. Scelta del file SIGNAL
4. Costi di transazione (opzionale)

---

### 4.2 Avvio con parametri CLI (non interattivo)

```bash
python scripts/report_strategia.py \
  --input /path/SIGNAL_xxx.csv \
  --signals-dir /path/segnali \
  --fee-bps 0.0 \
  --slippage-bps 0.0 \
  --initial-capital 100
```

Parametri principali:

| Parametro           | Descrizione                |
| ------------------- | -------------------------- |
| `--input`           | File SIGNAL diretto        |
| `--signals-dir`     | Cartella segnali           |
| `--fee-bps`         | Commissioni in bps         |
| `--slippage-bps`    | Slippage in bps            |
| `--initial-capital` | Capitale iniziale fallback |

---

## 5. Output

### 5.1 Report CSV

Viene generato un file:

```
REPORT_SIGNAL_<nome_file>.csv
```

Colonne principali:

```
Indicatore | Valore | Unità | Verificata
```

La colonna tecnica `Valore_raw` **non viene esportata**.

---

### 5.2 Ordine metriche

Il report viene stampato in **ordine logico validato**, ad esempio:

1. Equity Start
2. Equity End
3. Gross Profit
4. Gross Loss
5. Net Profit
6. Metriche Outlier (-Outlier)
7. Metriche operative
8. Metriche additive

L’ordine è forzato nel codice e non dipende dall’ordine di registrazione.

---

## 6. Metriche Outlier (-Outlier)

Il report include metriche **robustezza** che scartano trade eccezionali:

### Metodo

* Calcolo su `pnl_eur`
* Media: `μ`
* Volatilità: `σ = STDEV.P`
* Trade scartati se:

```
|pnl - μ| > 3 * σ
```

### Metriche

* `Gross Profit (-Outlier)`
* `Gross Loss (-Outlier)`
* `Net Profit (-Outlier)`
* `Outliers Removed (count)`
* `Outliers Removed (%)`

👉 Servono a validare la **stabilità della strategia** eliminando eventi eccezionali positivi e negativi.

---

## 7. Verifica consigliata

Dopo ogni run:

* `Net Profit ≈ Gross Profit + Gross Loss`
* `Outliers Removed (%) < 2%` → strategia stabile
* Differenza limitata tra `Net Profit` e `Net Profit (-Outlier)` → robustezza

---

## 8. Estensione metriche

Per aggiungere nuove metriche:

1. Creare file in `metrics/`
2. Usare `@register_metric`
3. Aggiungere il modulo in `_METRIC_MODULES`
4. Le metriche possono accettare:

   * `(equity_df, trades_df)` **oppure**
   * `(equity_df, trades_df, params)`

`apply_metrics()` gestisce entrambe.

---

## 9. Filosofia del progetto

* **Single source of truth numerica**
* Nessuna formattazione nelle metriche
* Report coerente con Excel
* Separazione netta tra:

  * calcolo
  * validazione
  * presentazione

---

## 10. Stato del progetto

✔ Produzione-ready
✔ Validazione robustezza
✔ Metriche additive e outlier-safe

---

Autore: Claudio
