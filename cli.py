#!/usr/bin/env python3
"""Interactive CLI for Episode Curator ReAct Agent."""

import argparse
import sys
from episode_curator import create_agent, EpisodeStore


MODELS = {
    "haiku": "claude-haiku-4-5-20251001",
    "sonnet": "claude-sonnet-4-20250514",
}


def main():
    parser = argparse.ArgumentParser(description="Episode Curator ReAct Agent CLI")
    parser.add_argument(
        "-m", "--model", default="haiku", choices=MODELS.keys(),
        help="Worker model (default: haiku, OAuth 支援；sonnet 需要 API key)",
    )
    parser.add_argument("--api-key", default=None, help="Anthropic API key")
    parser.add_argument("--threshold-pct", type=int, default=50,
                        help="壓縮門檻百分比 (預設 50%% of context window)")
    args = parser.parse_args()

    worker_model = MODELS[args.model]

    # Resolve context window and compute compression threshold
    from cli_app import get_model_context_window
    context_window = get_model_context_window(worker_model)
    threshold = context_window * args.threshold_pct // 100

    print("=" * 50)
    print("  Episode Curator ReAct Agent — Interactive CLI")
    print("=" * 50)
    print(f"  Worker: {args.model} ({worker_model})")
    print(f"  Context: {context_window // 1000}k | 壓縮門檻: {threshold // 1000}k ({args.threshold_pct}%)")
    print()
    print("指令：")
    print("  /episodes  — 查看已儲存的 episodes")
    print("  /facts     — 查看已提取的事實")
    print("  /tokens    — 查看 token 用量")
    print("  /quit      — 離開")
    print()

    agent = create_agent(
        worker_model=worker_model,
        curator_model="claude-haiku-4-5-20251001",
        threshold=threshold,
        max_iterations=15,
        api_key=args.api_key,
    )
    store = EpisodeStore()
    history = []

    while True:
        try:
            query = input("\033[36m你：\033[0m").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再見！")
            break

        if not query:
            continue

        if query == "/quit":
            print("再見！")
            break

        if query == "/episodes":
            idx = store._index
            if not idx:
                print("\033[33m（尚無 episodes）\033[0m\n")
            else:
                for eid, entry in sorted(idx.items()):
                    tags = ", ".join(entry.get("tags", []))
                    cont = f" (接續 #{entry['continues_episode']})" if entry.get("continues_episode") else ""
                    print(f"  \033[32m#{eid}\033[0m [{tags}] {entry['title']}{cont}")
                    print(f"        {entry['summary']}")
                print()
            continue

        if query == "/facts":
            facts = store.get_facts()
            if not facts:
                print("\033[33m（尚無事實）\033[0m\n")
            else:
                for f in facts:
                    print(f"  • {f}")
                print()
            continue

        if query == "/tokens":
            print(f"  累計對話長度：{len(history)} 條訊息")
            print(f"  Episodes 數量：{len(store._index)}")
            print(f"  Facts 數量：{len(store.get_facts())}")
            print()
            continue

        try:
            answer = agent.run(query, list(history))
            print(f"\033[33mAgent：\033[0m{answer}\n")
            history.append({"role": "user", "content": query})
            history.append({"role": "assistant", "content": answer})
        except Exception as e:
            print(f"\033[31m錯誤：{e}\033[0m\n")


if __name__ == "__main__":
    main()
