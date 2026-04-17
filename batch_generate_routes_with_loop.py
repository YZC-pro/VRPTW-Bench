import os
import json
import time
import re
from pathlib import Path
from typing import Any, Dict, Optional, List, Tuple

from dotenv import load_dotenv
from tqdm import tqdm

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from evaluate_solution import VRPTWEvaluator


# =========================
# 1. System Prompt
# =========================
SYSTEM_PROMPT = """ """


# =========================
# 2. Utils
# =========================
def build_instance_text(instance: Dict[str, Any]) -> str:
    parts = []
    parts.append(f'Instance name: {instance.get("name", "unknown")}')
    parts.append("")
    parts.append("Vehicle:")
    parts.append(json.dumps(instance["vehicle"], ensure_ascii=False, indent=2))
    parts.append("")
    parts.append("Depot:")
    parts.append(json.dumps(instance["depot"], ensure_ascii=False, indent=2))
    parts.append("")
    parts.append("Customers:")
    parts.append(json.dumps(instance["customers"], ensure_ascii=False, indent=2))
    return "\n".join(parts)


def extract_json_block(text: str) -> Optional[str]:
    text = text.strip()

    try:
        json.loads(text)
        return text
    except Exception:
        pass

    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.S)
    if m:
        candidate = m.group(1).strip()
        try:
            json.loads(candidate)
            return candidate
        except Exception:
            pass

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = text[start:end + 1]
        try:
            json.loads(candidate)
            return candidate
        except Exception:
            pass

    return None


def save_json(obj: Dict[str, Any], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def save_routes_pretty_linewise(result_obj: Dict[str, Any], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        f.write("{\n")
        f.write(f'  "case_name": {json.dumps(result_obj.get("case_name"), ensure_ascii=False)},\n')
        f.write(f'  "model_name": {json.dumps(result_obj.get("model_name"), ensure_ascii=False)},\n')
        f.write(f'  "attempt_count": {json.dumps(result_obj.get("attempt_count"))},\n')
        f.write(f'  "final_attempt": {json.dumps(result_obj.get("final_attempt"))},\n')
        f.write(f'  "final_prompt_type": {json.dumps(result_obj.get("final_prompt_type"), ensure_ascii=False)},\n')
        f.write(f'  "api_error": {json.dumps(result_obj.get("api_error"), ensure_ascii=False)},\n')
        f.write(f'  "json_parse_success": {json.dumps(result_obj.get("json_parse_success"))},\n')
        f.write('  "routes": [\n')

        routes = result_obj.get("routes") or []
        for i, route in enumerate(routes):
            comma = "," if i < len(routes) - 1 else ""
            f.write(f"    {json.dumps(route, ensure_ascii=False)}{comma}\n")

        f.write("  ]\n")
        f.write("}\n")

def parse_llm_routes(raw_text: str) -> Tuple[bool, Optional[Dict[str, Any]], Optional[List[List[int]]], Optional[str]]:
    if not raw_text:
        return False, None, None, "Empty response."

    json_block = extract_json_block(raw_text)
    if json_block is None:
        return False, None, None, "No valid JSON object found in response."

    try:
        parsed = json.loads(json_block)
    except Exception as e:
        return False, None, None, f"JSON load failed: {e}"

    if not isinstance(parsed, dict):
        return False, parsed, None, "Top-level JSON is not an object."

    if "routes" not in parsed:
        return False, parsed, None, "Missing key 'routes'."

    routes = parsed["routes"]
    if not isinstance(routes, list):
        return False, parsed, None, "'routes' is not a list."

    return True, parsed, routes, None


# =========================
# 3. Prompt loading / fallback
# =========================
def load_prompt_file(prompt_dir: Optional[Path], filename: str) -> Optional[str]:
    if prompt_dir is None:
        return None
    fp = prompt_dir / filename
    if fp.exists():
        return fp.read_text(encoding="utf-8").strip()
    return None


def get_initial_user_prompt(instance: Dict[str, Any], prompt_dir: Optional[Path] = None) -> str:

    template = load_prompt_file(prompt_dir, "initial_prompt.txt")
    if template is None:
        template = """ ///"""
    return template.format(instance_text=build_instance_text(instance))


def summarize_evaluation(eval_result: Dict[str, Any]) -> str:
    """
    把 evaluator 输出整理成更适合喂给 LLM 的错误摘要。
    """
    lines = []

    if not eval_result.get("parse_success", False):
        lines.append(f"- Parse failed: {eval_result.get('parse_error')}")

    if eval_result.get("exceeds_max_vehicles", False):
        lines.append(
            f"- Too many routes/vehicles: route_count={eval_result.get('route_count')}, "
            f"max_vehicles exceeded."
        )

    missing = eval_result.get("missing_customers", [])
    if missing:
        lines.append(f"- Missing customers: {missing}")

    duplicates = eval_result.get("duplicate_customers", [])
    if duplicates:
        lines.append(f"- Duplicate customers: {duplicates}")

    unknown_nodes = eval_result.get("unknown_nodes", [])
    if unknown_nodes:
        lines.append(f"- Unknown node ids: {unknown_nodes}")

    route_format_violations = eval_result.get("route_format_violations", [])
    if route_format_violations:
        lines.append("- Route format violations:")
        for v in route_format_violations[:10]:
            lines.append(f"  - route_index={v['route_index']}, reason={v['reason']}")

    capacity_violations = eval_result.get("capacity_violations", [])
    if capacity_violations:
        lines.append("- Capacity violations:")
        for v in capacity_violations[:10]:
            lines.append(
                f"  - route_index={v['route_index']}, load={v['load']}, "
                f"capacity={v['capacity']}, overload={v['overload']}"
            )

    tw_violations = eval_result.get("time_window_violations", [])
    if tw_violations:
        lines.append("- Time window violations:")
        for v in tw_violations[:10]:
            lines.append(
                f"  - route_index={v['route_index']}, from={v['from_node']}, to={v['to_node']}, "
                f"start_service={v['start_service']:.3f}, due={v['due']}, "
                f"lateness={v['lateness']:.3f}"
            )

    if not lines and eval_result.get("feasible", False):
        lines.append("- Solution is feasible.")

    return "\n".join(lines)


def get_repair_user_prompt(
    attempt_idx: int,
    instance: Dict[str, Any],
    previous_solution_obj: Optional[Dict[str, Any]],
    previous_raw_text: str,
    eval_result: Dict[str, Any],
    prompt_dir: Optional[Path] = None,
) -> str:
    """
    repair prompt

    """
    eval_summary = summarize_evaluation(eval_result)
    prev_solution_text = (
        json.dumps(previous_solution_obj, ensure_ascii=False, indent=2)
        if previous_solution_obj is not None
        else previous_raw_text
    )

    filename = f"repair_prompt_{attempt_idx}.txt"
    template = load_prompt_file(prompt_dir, filename)

    built_in_templates = {
        2: """///
""",
        3: """3
""",
        4: """4
""",
        5: """5
"""
    }

    if template is None:
        template = built_in_templates.get(attempt_idx, built_in_templates[5])

    return template.format(
        eval_summary=eval_summary,
        previous_solution=prev_solution_text,
        instance_text=build_instance_text(instance),
    )


# =========================
# 4. Single case loop
# =========================
def solve_one_case_with_loop(
    llm: ChatOpenAI,
    instance: Dict[str, Any],
    case_name: str,
    prompt_dir: Optional[Path],
    max_attempts: int = 5,
    distance_rounding: str = "none",
    sleep_sec: float = 1.0,
) -> Dict[str, Any]:
    evaluator = VRPTWEvaluator(instance, distance_rounding=distance_rounding)

    attempt_records = []
    best_result = None
    final_result = None

    previous_solution_obj = None
    previous_raw_text = ""

    for attempt_idx in range(1, max_attempts + 1):
        if attempt_idx == 1:
            user_prompt = get_initial_user_prompt(instance, prompt_dir)
            print(user_prompt)
            prompt_type = "initial"
        else:
            user_prompt = get_repair_user_prompt(
                attempt_idx=attempt_idx,
                instance=instance,
                previous_solution_obj=previous_solution_obj,
                previous_raw_text=previous_raw_text,
                eval_result=best_result["evaluation"],
                prompt_dir=prompt_dir,
            )
            prompt_type = f"repair_{attempt_idx}"

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ]

        api_error = None
        raw_text = ""
        parsed_obj = None
        routes = None
        json_ok = False
        parse_note = None

        try:
            response = llm.invoke(messages)
            raw_text = response.content if isinstance(response.content, str) else str(response.content)
        except Exception as e:
            api_error = str(e)

        if api_error is None:
            json_ok, parsed_obj, routes, parse_note = parse_llm_routes(raw_text)

        if json_ok and routes is not None:
            evaluation = evaluator.evaluate({"routes": routes})
        else:
            evaluation = evaluator.evaluate({"bad_response": raw_text})
            # evaluator 对这种对象会报 missing routes，这里保留更清晰一点的 note
            if not evaluation["parse_success"] and parse_note:
                evaluation["parse_error"] = parse_note

        record = {
            "attempt": attempt_idx,
            "prompt_type": prompt_type,
            "api_error": api_error,
            "json_parse_success": json_ok,
            "parse_note": parse_note,
            "raw_response": raw_text,
            "parsed_response": parsed_obj,
            "routes": routes,
            "evaluation": evaluation,
        }
        attempt_records.append(record)

        previous_solution_obj = parsed_obj
        previous_raw_text = raw_text
        best_result = record
        final_result = record

        print(
            f"    attempt={attempt_idx} | "
            f"prompt={prompt_type} | "
            f"api_ok={api_error is None} | "
            f"json_ok={json_ok} | "
            f"feasible={evaluation.get('feasible', False)}"
        )

        if evaluation.get("feasible", False):
            break

        if sleep_sec > 0 and attempt_idx < max_attempts:
            time.sleep(sleep_sec)

    summary = {
        "case_name": case_name,
        "feasible": final_result["evaluation"].get("feasible", False) if final_result else False,
        "attempt_count": len(attempt_records),
        "final_attempt": final_result["attempt"] if final_result else None,
        "final_prompt_type": final_result["prompt_type"] if final_result else None,
        "final_routes": final_result["routes"] if final_result else None,
        "final_evaluation": final_result["evaluation"] if final_result else None,
        "attempt_records": attempt_records,
    }
    return summary


# =========================
# 5. Batch runner
# =========================
def run_batch_with_loop(
    data_dir: str,
    output_dir: str,
    model_name: str,
    api_key: str,
    base_url: Optional[str] = None,
    prompt_dir: Optional[str] = None,
    temperature: float = 0.2,
    max_tokens: int = 4096,
    sleep_sec: float = 1.0,
    limit: Optional[int] = None,
    max_attempts: int = 5,
    distance_rounding: str = "none",
):
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    prompt_dir_path = Path(prompt_dir) if prompt_dir else None

    raw_dir = output_dir / "raw"
    routes_dir = output_dir / "routes"
    eval_dir = output_dir / "evaluations"
    logs_dir = output_dir / "attempt_logs"

    raw_dir.mkdir(parents=True, exist_ok=True)
    routes_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    llm_kwargs = {
        "model": model_name,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "api_key": api_key,
    }
    if base_url:
        llm_kwargs["base_url"] = base_url

    llm = ChatOpenAI(**llm_kwargs)

    files = sorted(data_dir.glob("*.json"))
    if limit is not None:
        files = files[:limit]

    if not files:
        print(f"No JSON files found in {data_dir}")
        return

    total = len(files)
    feasible_count = 0
    api_case_success = 0
    batch_start_time = time.time()

    for idx, fp in enumerate(tqdm(files, desc=f"Running {model_name} with loop"), start=1):
        case_start_time = time.time()
        case_name = fp.stem

        with open(fp, "r", encoding="utf-8") as f:
            instance = json.load(f)

        print(f"\n[{idx}/{total}] case={case_name}")

        summary = solve_one_case_with_loop(
            llm=llm,
            instance=instance,
            case_name=case_name,
            prompt_dir=prompt_dir_path,
            max_attempts=max_attempts,
            distance_rounding=distance_rounding,
            sleep_sec=sleep_sec,
        )

        attempt_records = summary["attempt_records"]
        final_record = attempt_records[-1]

        save_json(summary, logs_dir / f"{case_name}.json")
      
        raw_obj = {
            "case_name": case_name,
            "model_name": model_name,
            "final_attempt": summary["final_attempt"],
            "final_prompt_type": summary["final_prompt_type"],
            "raw_response": final_record["raw_response"],
            "api_error": final_record["api_error"],
        }
        save_json(raw_obj, raw_dir / f"{case_name}.json")

        # 保存最终 routes
        result_obj = {
            "case_name": case_name,
            "model_name": model_name,
            "attempt_count": summary["attempt_count"],
            "final_attempt": summary["final_attempt"],
            "final_prompt_type": summary["final_prompt_type"],
            "api_error": final_record["api_error"],
            "routes": final_record["routes"],
            "json_parse_success": final_record["json_parse_success"],
            "raw_response": final_record["raw_response"],
            "parsed_response": final_record["parsed_response"],
        }
        save_routes_pretty_linewise(result_obj, routes_dir / f"{case_name}.json")

        eval_obj = {
            "case_name": case_name,
            "model_name": model_name,
            "attempt_count": summary["attempt_count"],
            "feasible": summary["feasible"],
            "final_evaluation": summary["final_evaluation"],
        }
        save_json(eval_obj, eval_dir / f"{case_name}.json")

        if any(r["api_error"] is None for r in attempt_records):
            api_case_success += 1

        if summary["feasible"]:
            feasible_count += 1

        case_elapsed = time.time() - case_start_time
        total_elapsed = time.time() - batch_start_time

        print(
            f"[{idx}/{total}] {case_name} finished | "
            f"feasible={summary['feasible']} | "
            f"attempts={summary['attempt_count']} | "
            f"case_time={case_elapsed:.2f}s | "
            f"feasible_cases={feasible_count}/{idx} | "
            f"api_case_success={api_case_success}/{idx} | "
            f"total_time={total_elapsed:.2f}s"
        )

    print("\nBatch finished.")
    print(f"Model: {model_name}")
    print(f"Total cases: {total}")
    print(f"Feasible cases: {feasible_count}/{total}")
    print(f"Cases with at least one successful API call: {api_case_success}/{total}")
    print(f"Total elapsed time: {time.time() - batch_start_time:.2f}s")


# =========================
# 6. Main
# =========================
if __name__ == "__main__":
    load_dotenv()

    DATA_DIR = r""
    OUTPUT_DIR = r""
    PROMPT_DIR = r""   
    MODEL_NAME = ""

    API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
    BASE_URL = ""

    run_batch_with_loop(
        data_dir=DATA_DIR,
        output_dir=OUTPUT_DIR,
        model_name=MODEL_NAME,
        api_key=API_KEY,  
        base_url=BASE_URL,
        prompt_dir=PROMPT_DIR,
        temperature=1.0,
        max_tokens=4096,
        sleep_sec=1.0,
        limit=56,
        max_attempts=5,
        distance_rounding="none",
    )
