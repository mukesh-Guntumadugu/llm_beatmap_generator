# Qwen JSON Formatting Options

To use the **exact same prompt** for both models while getting structured JSON out of Qwen, we need to replicate what Gemini's `response_schema` feature does behind the scenes. 

Gemini's `response_schema` API does two things automatically: it injects JSON format instructions into the prompt invisibly, and it constrains the model's output tokens. 

For Qwen (running locally via Hugging Face `transformers`), we can replicate this using one of two approaches:

### Option 1: True API-Level Constraint (Logits Processing)
We can install an open-source library like **[lm-format-enforcer](https://github.com/noamgat/lm-format-enforcer)** or **[outlines](https://github.com/outlines-dev/outlines)**. 
- **How it works:** We pass our `Beat` Pydantic model to the library, and it hooks into Qwen's `model.generate()` loop. At every single letter it generates, it forces the probabilities of invalid JSON characters to 0%.
- **Advantage:** 100% guaranteed valid JSON output, and the prompt remains strictly identical to Gemini's with zero modifications.
- **Disadvantage:** Requires `pip install lm-format-enforcer` and slightly modifies the local inference code in `src/qwen_interface.py`.

### Option 2: "Hidden" Prompt Injection (The API Way)
When you pass `response_schema` to Gemini, Google's API silently appends JSON formatting instructions to your prompt before executing it. We can do the exact same thing in Python for Qwen.
- **How it works:** `src/beatmap_prompt.py` contains only the pure, shared prompt. Inside `run_qwen_fraxtil.py`, the Python script silently appends a "Please output as JSON using this schema" string *just before* sending it to Qwen. 
- **Advantage:** No extra libraries needed. From a research/prompting perspective, your "base prompt" is completely identical between the two models.
- **Disadvantage:** Relies on Qwen following the injected JSON instructions (though Qwen usually handles JSON well).

**Which approach would you prefer?** I can implement either one immediately. If you want the strictest equivalent to Gemini's API, Option 1 is the most accurate replication.
