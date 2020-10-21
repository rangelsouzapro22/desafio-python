import re


def highlight_search_word(doc_id, row_id, tokens, content):
    for token in tokens:
        without_case = re.compile(token, re.IGNORECASE)
        match_list = without_case.findall(content)
        for item in match_list:
            content = content.replace(item, f"\033[1;32;40m{item}\033[0;0m")
    return f"- document {doc_id}| row {row_id}: {content}"
