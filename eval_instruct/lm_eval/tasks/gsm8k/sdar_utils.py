"""SDAR's official OpenCompass GSM8K answer extraction and equivalence."""

import re


def _last_boxed(text):
    start = max(text.rfind("\\boxed"), text.rfind("\\fbox"))
    if start < 0:
        return None
    depth = 0
    for index in range(start, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


def _boxed_answer(text):
    boxed = _last_boxed(text)
    if boxed is None or not boxed.endswith("}"):
        return None
    for prefix in ("\\boxed{", "\\fbox{"):
        if boxed.startswith(prefix):
            answer = boxed[len(prefix) : -1]
            if answer.startswith("{") and answer.endswith("}"):
                answer = answer[1:-1]
            return answer
    return None


def _normalize_final_answer(answer):
    substitutions = (
        ("an ", ""),
        ("a ", ""),
        (".$", "$"),
        ("\\$", ""),
        (r"\ ", ""),
        (" ", ""),
        ("mbox", "text"),
        (",\\text{and}", ","),
        ("\\text{and}", ","),
        ("\\text{m}", "\\text{}"),
        ("\\le", "<"),
    )
    removed = (
        "square", "ways", "integers", "dollars", "mph", "inches", "ft",
        "hours", "km", "units", "\\ldots", "points", "feet", "minutes",
        "digits", "cents", "degrees", "cm", "gm", "pounds", "meters",
        "meals", "edges", "students", "multiples", "\\text{s}",
        "\\text{.}", "\\text{}", r"\mathrm{th}", r"^\circ", r"^{\circ}",
        r"\;", r",\!", "{,}", '"', "\\dots", "\n", "\r", "\f",
    )
    for before, after in substitutions:
        answer = answer.replace(before, after)
    for expression in removed:
        answer = answer.replace(expression, "")
    answer = re.sub(r"(\\text\{)\((.*?)\)(\})", r"\2", answer)
    answer = re.sub(r"(\\text\{|\\textbf\{|\\overline\{)(.*?)(\})", r"\2", answer)
    answer = re.sub(r"(\\boxed\{)(.*)(\})", r"\2", answer)
    final = re.findall(r"finalansweris(.*)", answer)
    if final:
        answer = final[-1]
    stated = re.findall(r"answer?is:?(.*)", answer)
    if stated:
        answer = stated[-1]
    math = re.findall(r"\$(.*?)\$", answer)
    if math:
        answer = math[-1]
    answer = answer.strip().replace("$", "")
    if "rac" in answer and "\\frac" not in answer:
        answer = answer.replace("rac", "\\frac")
    answer = re.sub(r"(frac)([^{])(.)", r"frac{\2}{\3}", answer)
    answer = re.sub(r"(sqrt)([^{])", r"sqrt{\2}", answer)
    if answer.replace(",", "").isdigit():
        answer = answer.replace(",", "")
    return answer


def _postprocess(text):
    boxed = _boxed_answer(text)
    if boxed:
        return boxed
    for sentence in text.split("."):
        if re.search(r"final answer|answer is", sentence.lower()):
            return _normalize_final_answer(sentence)
    return _normalize_final_answer(text.split(".")[0])


def _fix_fracs(text):
    parts = text.split("\\frac")
    fixed = parts[0]
    for part in parts[1:]:
        fixed += "\\frac"
        if part.startswith("{"):
            fixed += part
        elif len(part) >= 2:
            fixed += "{" + part[0] + "}{" + part[1] + "}" + part[2:]
        else:
            return text
    return fixed


def _strip(text):
    text = str(text).strip().replace("\n", "").rstrip(".")
    for old, new in (
        ("\\!", ""), ("\\ ", ""), ("\\\\", "\\"),
        ("tfrac", "frac"), ("dfrac", "frac"), ("\\left", ""),
        ("\\right", ""), ("^{\\circ}", ""), ("^\\circ", ""),
        ("\\$", ""), ("$", ""), ("\\text", ""), ("x\\in", ""),
        ("\\%", ""), ("%", ""), ("\\cdot", ""), ("\\mathbf", ""),
    ):
        text = text.replace(old, new)
    without_unit = re.sub(r"\\text{.*?}$", "", text).strip()
    if without_unit:
        text = without_unit
    text = re.sub(r"\\mbox{.*?}", "", text)
    text = re.sub(r"(\d+)\.0+([^\d])", r"\1\2", text)
    text = re.sub(r"(\d+)\.0+$", r"\1", text)
    text = text.replace(" .", " 0.").replace("{.", "{0.")
    if text.startswith("."):
        text = "0" + text
    if len(text.split("=")) == 2 and len(text.split("=")[0]) <= 2:
        text = text.split("=")[1]
    text = re.sub(r"\\sqrt(\w+)", r"\\sqrt{\1}", text).replace(" ", "")
    text = _fix_fracs(text)
    parts = text.split("/")
    if len(parts) == 2:
        try:
            left, right = int(parts[0]), int(parts[1])
            if text == f"{left}/{right}":
                text = f"\\frac{{{left}}}{{{right}}}"
        except ValueError:
            pass
    return text


def _is_equiv(prediction, reference):
    if prediction is None or reference is None:
        return prediction is reference
    try:
        left, right = _strip(prediction), _strip(reference)
        if left == right:
            return True
        if _normalize_final_answer(left) == _normalize_final_answer(right):
            return True
    except Exception:
        pass
    try:
        return _normalize_final_answer(prediction) == _normalize_final_answer(reference)
    except Exception:
        return prediction == reference


def process_results(doc, results):
    prediction = _postprocess(results[0])
    reference = doc["answer"].split("#### ")[-1].replace(",", "")
    return {"exact_match": int(_is_equiv(prediction, reference))}
