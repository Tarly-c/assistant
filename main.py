from __future__ import annotations

import argparse

import uvicorn

from medical_assistant.config import get_settings
from medical_assistant.graph.workflow import build_workflow
from medical_assistant.schemas.response import AssistantResponse
from medical_assistant.schemas.state import ConversationState
from medical_assistant.storage.session_store import InMemorySessionStore


def run_cli() -> None:
    workflow = build_workflow()
    store = InMemorySessionStore()
    record = store.create()
    session_id = record.session_id

    print("Medical Assistant CLI 已启动。输入 exit 退出。")
    print(f"session_id={session_id}")

    while True:
        try:
            question = input("\n[User]> ").strip()
        except KeyboardInterrupt:
            print("\n退出。")
            break

        if not question:
            continue
        if question.lower() == "exit":
            break

        record = store.get(session_id)
        result = workflow.invoke(
            {
                "question": question,
                "conversation_state": record.state.model_dump(mode="json"),
            }
        )

        updated_state = ConversationState.model_validate(
            result.get("conversation_state") or record.state.model_dump(mode="json")
        )
        response = AssistantResponse.model_validate(
            result.get("response") or {"response_type": "answer", "content": ""}
        )

        store.save(session_id, updated_state)
        store.append_message(
            session_id,
            {
                "user": question,
                "assistant": response.content,
                "response_type": response.response_type,
                "state": updated_state.snapshot(),
            },
        )

        print("\n[Assistant]")
        print(response.content)
        print("\n[State]")
        print(updated_state.snapshot())


def run_api() -> None:
    settings = get_settings()
    uvicorn.run(
        "medical_assistant.api.web_server:web_app",
        host=settings.api_host,
        port=settings.api_port,
        reload=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode",
        nargs="?",
        choices=["cli", "api"],
        default="cli",
        help="cli 或 api",
    )
    args = parser.parse_args()

    if args.mode == "api":
        run_api()
    else:
        run_cli()


if __name__ == "__main__":
    main()
