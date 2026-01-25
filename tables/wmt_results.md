# WMT17 German-English Translation Results

## Model Configuration

| Parameter | Value |
|-----------|-------|
| d_model | 64 |
| n_heads | 4 |
| num_encoder_layers | 2 |
| num_decoder_layers | 2 |
| dim_feedforward | 128 |
| max_len | 64 |

## Evaluation Results

| Metric | Value |
|--------|-------|
| BLEU Score | 0.0205 |
| BLEU-1 | 0.2273 |
| BLEU-2 | 0.0432 |
| BLEU-3 | 0.0074 |
| BLEU-4 | 0.0024 |
| Test Samples | 100 |

## Sample Translations

### Example 1
- **Source (DE):** 28-jähriger Koch in San Francisco Mall tot aufgefunden
- **Reference (EN):** 28-Year-Old Chef Found Dead at San Francisco Mall
- **Prediction:** The Social Minister has been held in the Middle East-southis-southis.

### Example 2
- **Source (DE):** Ein 28-jähriger Koch, der vor kurzem nach San Francisco gezogen ist, wurde im Treppenhaus eines örtlichen Einkaufzentrums tot aufgefunden.
- **Reference (EN):** A 28-year-old chef who had recently moved to San Francisco was found dead in the stairwell of a local mall this week.
- **Prediction:** A number of 28, the former Minister was recently recently recently recently in the former Minister, the European Investment Bank has been held in a special community.

### Example 3
- **Source (DE):** Der Bruder des Opfers sagte aus, dass er sich niemanden vorstellen kann, der ihm schaden wollen würde, Endlich ging es bei ihm wieder bergauf.
- **Reference (EN):** But the victims brother says he cant think of anyone who would want to hurt him, saying, Things were finally going well for him.
- **Prediction:** The former Minister said that he has been able to be able to see the fact that he could be able to be able to be able to be able to be able to keep it.

### Example 4
- **Source (DE):** Der am Mittwoch morgen in der Westfield Mall gefundene Leichnam wurde als der 28 Jahre alte Frank Galicia aus San Francisco identifiziert, teilte die gerichtsmedizinische Abteilung in San Francisco mit.
- **Reference (EN):** The body found at the Westfield Mall Wednesday morning was identified as 28-year-old San Francisco resident Frank Galicia, the San Francisco Medical Examiners Office said.
- **Prediction:** The vote will be taken in the end of the BSE and the old old old old old old-souths, as a number of years ago, which has been made in the former Minister of the former Minister.

### Example 5
- **Source (DE):** Das San Francisco Police Department sagte, dass der Tod als Mord eingestuft wurde und die Ermittlungen am Laufen sind.
- **Reference (EN):** The San Francisco Police Department said the death was ruled a homicide and an investigation is ongoing.
- **Prediction:** The former Minister said that the death of the war has been made as a serious risk of the Russian authorities.

### Example 6
- **Source (DE):** Der Bruder des Opfers, Louis Galicia, teilte dem ABS Sender KGO in San Francisco mit, dass Frank, der früher als Koch in Boston gearbeitet hat, vor sechs Monaten seinen Traumjob als Koch im Sons & Daughters Restaurant in San Francisco ergattert hatte.
- **Reference (EN):** The victims brother, Louis Galicia, told ABC station KGO in San Francisco that Frank, previously a line cook in Boston, had landed his dream job as line chef at San Franciscos Sons & Daughters restaurant six months ago.
- **Prediction:** The former Minister, the fact of the European Investment Bank, the former Minister, the former Minister, has been held in the former Minister, as a number of months ago, as a number of months ago, as a number of former Minister, he has been held in the past, and his colleague, as long

### Example 7
- **Source (DE):** Ein Sprecher des Sons & Daughters sagte, dass sie über seinen Tod schockiert und am Boden zerstört seien.
- **Reference (EN):** A spokesperson for Sons & Daughters said they were shocked and devastated by his death.
- **Prediction:** A former Minister said, the fact that it has been said that it has been destroyed and the war.

### Example 8
- **Source (DE):** Wir sind ein kleines Team, das wie eine enge Familie arbeitet und wir werden ihn schmerzlich vermissen, sagte der Sprecher weiter.
- **Reference (EN):** We are a small team that operates like a close knit family and he will be dearly missed, the spokesperson said.
- **Prediction:** We are a small small-sized and a small-sized and we are doing a great deal of the oil industry.

### Example 9
- **Source (DE):** Unsere Gedanken und unser Beileid sind in dieser schweren Zeit bei Franks Familie und Freunden.
- **Reference (EN):** Our thoughts and condolences are with Franks family and friends at this difficult time.
- **Prediction:** Our approach and our family is the right of the family and children.

### Example 10
- **Source (DE):** Louis Galicia gab an, dass Frank zunächst in Hostels lebte, aber dass, die Dinge für ihn endlich bergauf gingen.
- **Reference (EN):** Louis Galicia said Frank initially stayed in hostels, but recently, Things were finally going well for him.
- **Prediction:** The former Minister has been able to be able to start by the fact that the armed forces, but the way of the issues of the past.

## Error Analysis

| Category | Count | Percentage |
|----------|-------|------------|
| Empty Output | 0 | 0.0% |
| Repetition | 12 | 12.0% |
| Truncated | 0 | 0.0% |
| Reasonable | 88 | 88.0% |