# Beatmap CSV Format Documentation

Our Language Models (Qwen and Gemini) generate beatmap data in a specific 7-column CSV format. Here is the strict definition for each field.

## Columns

| Column Name      | Data Type | Description                                                                                                                                           | Example    |
| :--------------- | :-------- | :---------------------------------------------------------------------------------------------------------------------------------------------------- | :--------- |
| `time_ms`        | Float     | The exact timestamp of the note in milliseconds, relative to the start of the song.                                                                   | `1250.0`   |
| `beat_position`  | Float     | The beat number from the start of the song (e.g. beat 1.0, 1.25, 2.5).                                                                                | `3.25`     |
| `notes`          | String    | A 4-character string representing the 4 pads (Left, Down, Up, Right). See "Note Types" below. For measure separators, this is exactly `","`.          | `"1000"`   |
| `placement_type` | Integer   | The rhythmic categorization of the note: `0`=unsure/empty, `1`=onset, `2`=beat, `3`=grid, `4`=percussive, `5`=unaligned. Separator rows use `-1`. | `4`        |
| `note_type`      | Integer   | The rhythmic subdivision: `0`=whole, `1`=half, `2`=quarter, `3`=eighth, `4`=extended (16th+). Separator rows use `-1`.                            | `2`        |
| `confidence`     | Float     | The model's confidence in this placement (0.0 to 1.0).                                                                                                | `0.95`     |
| `instrument`     | String    | The dominant instrument detected (e.g. `kick`, `snare`, `bass`, `melody`, `unknown`). Separator rows use `separator`.                                 | `kick`     |

## Note Types (`notes` column)

The `notes` string is always 4 characters long (one for each directional pad: L,D,U,R). The characters can be:

*   `0`: Empty slot (no note to step on).
*   `1`: Standard tap note.
*   `2`: Hold note **HEAD** (Start stepping and hold down).
*   `3`: Hold note **TAIL** (Release the held note).

**Hold Note Example:**
If the player must hold the Left arrow down from time `500.0` to `1000.0`, the CSV will look like:
```csv
500.0,2.0,2000,4,2,1.0,bass      <-- Start holding Left (2)
625.0,2.25,0000,0,3,1.0,unknown  <-- (Still holding...)
750.0,2.5,0100,4,3,0.82,snare    <-- (Still holding... also tap Down (1))
875.0,2.75,0000,0,3,1.0,unknown  <-- (Still holding...)
1000.0,3.0,3000,4,2,0.91,kick    <-- Release Left (3)
```

## Measure Separators

StepMania files group notes into structural "measures" (usually 4 beats). To signify the end of a measure to the parser, every measure must end with a separator row. A separator row always has exactly these values:

```csv
time_ms,beat_position,",",-1,-1,1.0,separator
```
*(Note: Because the `notes` column is literally just a comma `,`, it often appears enclosed in quotes like `","` depending on the CSV dialect mapping).*
