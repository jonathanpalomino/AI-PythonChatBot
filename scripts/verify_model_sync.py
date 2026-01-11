import asyncio
import os
import sys
from unittest.mock import AsyncMock, MagicMock

# Add src to path
sys.path.append(os.getcwd())

from src.models.llm_models import LLMModel
from src.providers.manager import provider_manager, ProviderType, ModelInfo, ModelType


# Mock provider for testing
class MockLocalProvider:
    def get_available_models(self):
        return [
            ModelInfo(
                name="gemma3",
                provider=ProviderType.LOCAL,
                context_window=8192,
                supports_function_calling=False,
                supports_streaming=True,
                model_type=ModelType.CHAT  # Initially inferred as CHAT
            ),
            ModelInfo(
                name="deepseek-r1",
                provider=ProviderType.LOCAL,
                context_window=8192,
                supports_function_calling=False,
                supports_streaming=True,
                model_type=ModelType.CHAT  # Initially inferred as CHAT
            )
        ]


async def verify_logic():
    print("--- Verifying Logic with Mocks ---")

    # Mock DB Session
    mock_session = AsyncMock()

    # Inject mock provider
    provider_manager.providers[ProviderType.LOCAL] = MockLocalProvider()

    # 1. Test Sync: No existing models
    print("\n[Test 1] Sync new models...")
    # Mock execute result for "existing model" check -> returns empty (None)
    mock_result_empty = MagicMock()
    mock_result_empty.scalar_one_or_none.return_value = None
    mock_session.execute.return_value = mock_result_empty

    await provider_manager.sync_available_models(mock_session)

    # Check what was added
    added_models = [call.args[0] for call in mock_session.add.call_args_list]
    print(f"Models added: {len(added_models)}")

    for m in added_models:
        print(f"Added Model: {m.model_name}, Type: {m.model_type}")
        if m.model_name == "deepseek-r1":
            if m.model_type == "reasoning":
                print("✅ deepseek-r1 correctly inferred as REASONING via heuristic")
            else:
                print(f"❌ deepseek-r1 failed heuristic check. Got: {m.model_type}")

        if m.model_name == "gemma3":
            if m.model_type == "chat":
                print("✅ gemma3 defaulted to CHAT (Expected)")
            else:
                print(f"❌ gemma3 should be CHAT. Got: {m.model_type}")

    # 2. Test Sync: Existing model with user override
    print("\n[Test 2] Sync existing model with user override...")
    mock_session.reset_mock()

    # Mock existing gemma3 with IS_CUSTOM=True and REASONING type
    existing_gemma = LLMModel(
        provider="local",
        model_name="gemma3",
        model_type="reasoning",  # User set this manually
        is_custom=True
    )

    # Mock execute result
    mock_result_existing = MagicMock()
    mock_result_existing.scalar_one_or_none.return_value = existing_gemma
    mock_session.execute.return_value = mock_result_existing

    await provider_manager.sync_available_models(mock_session)

    # Check that model_type was NOT updated (since is_custom=True)
    print(f"Existing Model Type after sync: {existing_gemma.model_type}")
    if existing_gemma.model_type == "reasoning":
        print("✅ User override respected (kept as REASONING)")
    else:
        print("❌ User override overwritten!")

    # 3. Test Sync: Upgrade CHAT to REASONING automatically
    print("\n[Test 3] Sync upgrade CHAT to REASONING...")
    mock_session.reset_mock()

    # Mock existing deepseek-r1 as CHAT (from old sync)
    existing_r1 = LLMModel(
        provider="local",
        model_name="deepseek-r1",
        model_type="chat",
        is_custom=False
    )

    mock_result_upgrade = MagicMock()
    mock_result_upgrade.scalar_one_or_none.return_value = existing_r1
    mock_session.execute.return_value = mock_result_upgrade

    await provider_manager.sync_available_models(mock_session)

    print(f"Existing R1 Type after sync: {existing_r1.model_type}")
    if existing_r1.model_type == "reasoning":
        print("✅ Auto-upgrade from CHAT to REASONING worked")
    else:
        print("❌ Auto-upgrade failed")


if __name__ == "__main__":
    asyncio.run(verify_logic())
