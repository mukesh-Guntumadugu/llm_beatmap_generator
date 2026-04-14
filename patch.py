import re
with open('scripts/train_hierarchical_mumu.py', 'r') as f:
    text = f.read()
import traceback

p1 = """                except Exception as e:
                    print(f"\\n  [SKIP] batch {num_batches}: {e}\\n")"""
p2 = """                except Exception as e:
                    import traceback
                    print(f"\\n  [SKIP] batch {num_batches}: {e}\\n{traceback.format_exc()}\\n")"""

text = text.replace(p1, p2)
with open('scripts/train_hierarchical_mumu.py', 'w') as f:
    f.write(text)
