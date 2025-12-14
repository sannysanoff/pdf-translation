#!/usr/bin/env python3
"""
Script to translate strings in JSON file to Russian using LLM.
Renames original file to .backup if it doesn't exist, otherwise fails.
"""

import json
import os
import sys
import openai

#MODEL = "mistralai/devstral-2512:free"
#MODEL = "amazon/nova-2-lite-v1:free"
MODEL = "amazon/nova-2-lite-v1"

def load_cache():
    if os.path.exists('translate_cache.json'):
        with open('translate_cache.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def save_cache(cache):
    with open('translate_cache.json', 'w', encoding='utf-8') as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)

def translate_strings_llm(strings):
    if not strings:
        return [], True
    try:
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            print("Error: OPENROUTER_API_KEY environment variable not set")
            return strings, False
        client = openai.OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key
        )
        prompt = "Пожалуйста переведи весь этот текст на годный русский язык в стиле детского школьного учебника - консервативный русский язык. Response must be valid JSON! "
        data = {f"K{i+1:03d}": s for i, s in enumerate(strings)}
        messages = [
            {"role": "user", "content": prompt + "\n\n" + json.dumps(data, ensure_ascii=False)}
        ]
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages
        )
        content = response.choices[0].message.content
        start = content.find('{')
        end = content.rfind('}') + 1
        if start == -1 or end == 0:
            print(f"No JSON found in response. Full response: {content}")
            return strings, False
        json_str = content[start:end]
        translated_data = json.loads(json_str)
        return [translated_data[f"K{i+1:03d}"] for i in range(len(strings))], True
    except Exception as e:
        print(f"Translation failed: {e}")
        return strings, False

def translate_strings(strings):
    cache = load_cache()
    to_translate = []
    indices = []
    for i, s in enumerate(strings):
        if s not in cache:
            to_translate.append(s)
            indices.append(i)
    if to_translate:
        translated_new, success = translate_strings_llm(to_translate)
        if not success:
            return None
        for idx, trans in zip(indices, translated_new):
            cache[strings[idx]] = trans
    save_cache(cache)
    return [cache[s] for s in strings]

def collect_strings(obj):
    strings = []
    if isinstance(obj, dict):
        for v in obj.values():
            strings.extend(collect_strings(v))
    elif isinstance(obj, list):
        for item in obj:
            strings.extend(collect_strings(item))
    elif isinstance(obj, str):
        strings.append(obj)
    return strings

def replace_strings(obj, translated_iter):
    if isinstance(obj, dict):
        for k in obj:
            obj[k] = replace_strings(obj[k], translated_iter)
    elif isinstance(obj, list):
        for j in range(len(obj)):
            obj[j] = replace_strings(obj[j], translated_iter)
    elif isinstance(obj, str):
        obj = next(translated_iter)
    return obj

def main():
    if len(sys.argv) != 2:
        print("Usage: python from_unicode.py <file_path>")
        sys.exit(1)

    file_path = sys.argv[1]

    if not os.path.exists(file_path):
        print(f"Error: File {file_path} does not exist")
        sys.exit(1)

    backup_path = file_path + '.backup'
    if os.path.exists(backup_path):
        print(f"Error: Backup file {backup_path} already exists")
        sys.exit(1)

    # Read from file
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Collect all strings
    strings = collect_strings(data)

    # Translate strings
    translated = translate_strings(strings)

    if translated is None:
        print("Translation failed, not saving")
        return

    # Replace strings in data
    translated_iter = iter(translated)
    data = replace_strings(data, translated_iter)

    # First, rename to backup
    os.rename(file_path, backup_path)

    # Write to original path with proper UTF-8
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"Translated {file_path}, original backed up as {backup_path}")


if __name__ == '__main__':
    main()
