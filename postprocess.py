import json
import os
import re


# Extract and validate HOI output
def extract_and_validate_hoi_output(text: str):
    """Extract HOI data from model output with validation"""
    # Extract content between <answer> tags
    if '<answer>' not in text:
        if '</think>' in text:
            text_split = text.split('</think>')
            # add <answer> after </think>
            text = text_split[0] + '</think><answer>' + text_split[1]
        if not "</think>" in text:
            text = '<answer>' + text
    if '</answer>' not in text:
        text += '</answer>'

    answer_match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)
    if answer_match:
        json_str = answer_match.group(1).strip()
    else:
        json_match = re.search(r'```json(.*?)```', text, re.DOTALL)
        if json_match:
            json_str = json_match.group(1).strip()
        else:
            print("Error: No <answer> tags found")
            return None
    
    # Clean JSON string
    json_str = json_str.replace("'", '"')  # Convert single to double quotes
    json_str = re.sub(r'//.*?\n', '', json_str)  # Remove comments
    json_str = re.sub(r'#.*?\n', '', json_str)  # Remove comments
    json_str = re.sub(r',\s*]', ']', json_str)  # Fix trailing commas
    json_str = re.sub(r',\s*}', '}', json_str)  # Fix trailing commas
    
    # Handle verb class format issues
    json_str = re.sub(
        r'"verb class":\s*"([^"]+)"', 
        r'"verb class": "\1"', 
        json_str
    )
    
    # Handle list-formatted verb classes
    json_str = re.sub(
        r'"verb class":\s*\[([^\]]+)\]', 
        r'"verb class": "\1"', 
        json_str
    )
    
    try:
        data = json.loads(json_str)
        
        # Validate top-level structure
        if not isinstance(data, list):
            raise ValueError("Top-level structure must be a list")
            
        valid_hois = []
        for item in data:
            if not isinstance(item, dict):
                continue
            
            # Check required keys
            required_keys = ['human', 'object', 'object class', 'verb class']
            if not all(key in item for key in required_keys):
                continue
                
            # Validate bounding box formats
            for bbox_type in ['human', 'object']:
                bbox = item[bbox_type]
                if not isinstance(bbox, list) or len(bbox) != 4:
                    continue
                    
            # Ensure verb class is string
            if not isinstance(item['verb class'], str):
                item['verb class'] = str(item['verb class'])
                
            valid_hois.append(item)
            
        return valid_hois
        
    except (json.JSONDecodeError, ValueError) as e:
        print(f"JSON parsing error: {e}")
        return []