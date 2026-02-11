import re
import argparse
from pathlib import Path

def integrate_chart(ssc_path, generated_notes_path, output_path=None, target_difficulty="Easy", target_stepstype="dance-single"):
    """
    Replaces the #NOTES section of a specific chart in an SSC file with generated notes.
    """
    print(f"Reading SSC: {ssc_path}")
    with open(ssc_path, 'r', encoding='utf-8') as f:
        content = f.read()
        
    print(f"Reading Generated Notes: {generated_notes_path}")
    with open(generated_notes_path, 'r', encoding='utf-8') as f:
        new_notes = f.read().strip()
        
    # Ensure new notes end with a semicolon if not present? 
    # The generated file ends with ';', but let's be safe.
    if not new_notes.strip().endswith(';'):
        new_notes += ";"
        
    # Logic:
    # 1. Split file into charts by "#NOTEDATA:;"
    # 2. Iterate and find the target chart
    # 3. Replace keys in that chart
    
    charts = content.split("#NOTEDATA:;")
    
    # The first chunk is the global header (before the first #NOTEDATA)
    # But wait, #NOTEDATA:; is the delimiter start.
    # So charts[0] is the global header. charts[1:] are the charts.
    
    new_charts = [charts[0]] # Start with header
    
    found_target = False
    
    for chart in charts[1:]: 
        # Check if this is the target
        if f"#STEPSTYPE:{target_stepstype};" in chart and f"#DIFFICULTY:{target_difficulty};" in chart:
            print(f"Found Target Chart: {target_stepstype} - {target_difficulty}")
            found_target = True
            
            # Replace #NOTES section
            # Regex to match #NOTES:\s*<content>;
            # We use likely DOTALL to match newlines
            
            # Escape regex special chars if any? #NOTES: should be safe.
            # Capture everything from #NOTES: up to the next semicolon.
            # Warning: Semicolons might be in comments? But usually #NOTES data ends with a semicolon.
            
            pattern = r"(#NOTES:)(.*?)(;)"
            
            # We want to replace group 2 (content) with new_notes
            # But re.sub is tricky with groups.
            # Easier: replace the whole block
            
            # Construct replacement string. 
            # Note: new_notes already has the trailing semicolon from the generation script (or we added it).
            # So if new_notes has ';', we shouldn't add another one in the pattern.
            
            # Let's clean new_notes to be just the data, no trailing semicolon for safety in substitution logic
            clean_notes_data = new_notes.rstrip(';')
            
            replacement = f"\\1\n{clean_notes_data}\n\\3" 
            
            # Apply regex
            new_chart_content = re.sub(pattern, replacement, chart, count=1, flags=re.DOTALL)
            new_charts.append(new_chart_content)
            
        else:
            new_charts.append(chart)
            
    
    if not found_target:
        print(f"Target chart ({target_stepstype}/{target_difficulty}) not found! Creating new chart.")
        clean_notes_data = new_notes.rstrip(';')
        new_chart = f"""
#CHARTNAME:;
#STEPSTYPE:{target_stepstype};
#DESCRIPTION:Generated;
#CHARTSTYLE:;
#DIFFICULTY:{target_difficulty};
#METER:1;
#RADARVALUES:0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0;
#CREDIT:AI;
#NOTES:
{clean_notes_data}
;
"""
        new_charts.append(new_chart)
    else:
        print("Integration successful in memory.")
        
    # Join back
    final_content = "#NOTEDATA:;".join(new_charts)
    
    if output_path:
        save_path = output_path
    else:
        save_path = ssc_path # Overwrite
        
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(final_content)
        
    print(f"Saved integrated file to: {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("ssc_path", help="Original SSC file")
    parser.add_argument("generated_notes", help="File containing generated notes")
    parser.add_argument("--output", help="Output path (optional, defaults to overwriting ssc_path)")
    
    args = parser.parse_args()
    
    integrate_chart(args.ssc_path, args.generated_notes, args.output)
