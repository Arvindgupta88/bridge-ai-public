from src.core.logger import logger
from src.core.config_validation import validate_configuration
from src.core.readiness import check_readiness
from src.core.settings import (
    APP_NAME,
    APP_DESCRIPTION,
    APP_VERSION,
    PAGE_ICON,
    PAGE_LAYOUT,
    CHAT_PROVIDER,
    OLLAMA_CHAT_MODEL,
)
from src.llm.providers.exceptions import ChatProviderError

from src.llm.provider_catalog import (
    PROVIDER_CATALOG,
    ProviderName,
)

from src.rag.pipeline import RagPipeline
from src.services.chat_service import ChatService
from src.services.startup_service import StartupService
from src.rag.retrieval_mode import RetrievalMode
from src.prompts.prompt_profile import PromptProfile
from src.ui.source_formatter import build_source_references
import hashlib

import streamlit as st

def run_app():

    logger.info("Bridge AI started")

    st.set_page_config(
        page_title=APP_NAME,
        page_icon=PAGE_ICON,
        layout=PAGE_LAYOUT,
    )

    configuration = validate_configuration()
    if not configuration.is_valid:
        logger.error("Application configuration validation failed")
        for issue in configuration.issues:
            if issue.level == "ERROR":
                st.error(issue.message)
        st.stop()

    # Create conversation memory
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "document_indexed" not in st.session_state:
        st.session_state.document_indexed = False

    if "document_name" not in st.session_state:
        st.session_state.document_name = None
    if "document_hash" not in st.session_state:
        st.session_state.document_hash = None
    if "document_error" not in st.session_state:
        st.session_state.document_error = None
    if "document_uploader_key" not in st.session_state:
        st.session_state.document_uploader_key = 0

    if "provider_name" not in st.session_state:
        st.session_state.provider_name = CHAT_PROVIDER
    if "model_name" not in st.session_state:
        st.session_state.model_name = OLLAMA_CHAT_MODEL

    available_provider_names = [
        provider_name.value
        for provider_name, definition in PROVIDER_CATALOG.items()
        if definition.is_available()
    ]
    if st.session_state.provider_name not in available_provider_names:
        st.session_state.provider_name = ProviderName.OLLAMA.value
    selected_definition = PROVIDER_CATALOG[
        ProviderName(st.session_state.provider_name)
    ]
    if st.session_state.model_name not in selected_definition.models:
        st.session_state.model_name = selected_definition.models[0]

    selected_configuration = validate_configuration(
        provider_name=st.session_state.provider_name,
        model=st.session_state.model_name,
    )
    if not selected_configuration.is_valid:
        logger.error("Selected application configuration is invalid")
        for issue in selected_configuration.issues:
            if issue.level == "ERROR":
                st.error(issue.message)
        st.stop()

    if "rag_pipeline" not in st.session_state:
        try:
            st.session_state.rag_pipeline = RagPipeline()
        except Exception as error:
            logger.error(
                "Application service construction failed (error_type=%s)",
                type(error).__name__,
            )
            st.error(
                "Bridge AI could not initialize its local search service. "
                "Check the application logs and retry."
            )
            st.stop()

    pipeline = st.session_state.rag_pipeline

    if "startup_service" not in st.session_state:
        startup_service = None
        try:
            startup_service = StartupService(
                pipeline=pipeline
            )
            knowledge_sync_summary = startup_service.initialize()
        except Exception as error:
            logger.error(
                "Startup knowledge synchronization failed (error_type=%s)",
                type(error).__name__,
            )
            knowledge_sync_summary = {
                "skip": 0,
                "indexed": 0,
                "deleted": 0,
                "reindexed": 0,
                "failed": 1,
                "metadata_synced": False,
                "errors": [{
                    "action": "scan",
                    "error": "Knowledge synchronization could not start.",
                }],
            }
        if startup_service is not None:
            st.session_state.startup_service = startup_service
        st.session_state.knowledge_sync_summary = knowledge_sync_summary
        st.session_state.initialization_complete = True


    if "chat_service" not in st.session_state:
        st.session_state.chat_service = ChatService(
            pipeline = pipeline,
        )

    chat_service = st.session_state.chat_service

    readiness = check_readiness(
        provider_name=st.session_state.get("provider_name", CHAT_PROVIDER),
        model=st.session_state.get("model_name", OLLAMA_CHAT_MODEL),
        initialization_complete=st.session_state.get(
            "initialization_complete", False
        ),
    )

    if "retrieval_mode" not in st.session_state:
        st.session_state.retrieval_mode = RetrievalMode.GENERAL

    if "prompt_profile" not in st.session_state:
        st.session_state.prompt_profile = (PromptProfile.BANKING_MODERNIZATION)

    with st.sidebar:
        st.header("Bridge AI")
        st.subheader("AI Model")
        available_provider_names = [
            provider_name.value
            for provider_name, definition
            in PROVIDER_CATALOG.items()
            if definition.is_available()
        ]

        if (
            st.session_state.provider_name
            not in available_provider_names
        ):
            st.session_state.provider_name = (
                ProviderName.OLLAMA.value
            )

        st.selectbox(
            "Provider",
            options = available_provider_names,
            key = "provider_name",
            format_func=lambda provider_name: (
                PROVIDER_CATALOG[
                    ProviderName(provider_name)
                ].display_name
            ),
        )

        selected_provider = PROVIDER_CATALOG[
            ProviderName(st.session_state.provider_name)
        ]

        if (
            st.session_state.model_name
            not in selected_provider.models
        ):
            st.session_state.model_name = (
                selected_provider.models[0]
            )

        st.selectbox(
            "Model",
            options=selected_provider.models,
            key="model_name",
        )

        if (
            selected_provider
            .credential_environment_variable
            is not None
        ):
            st.warning(
                "Remote provider selected. Prompts, conversation "
                "history, and retrieved document context may be "
                "sent outside this computer."
            )

        unavailable_providers = [
            definition
            for definition in PROVIDER_CATALOG.values()
            if not definition.is_available()
        ]

        if unavailable_providers:
            st.caption("Unavailable providers")

            for definition in unavailable_providers:
                reason = definition.unavailable_reason()

                st.button(
                    f"{definition.display_name} — {reason}",
                    key=(
                        "unavailable_provider_"
                        f"{definition.name.value}"
                    ),
                    disabled=True,
                    use_container_width=True,
                )

        st.subheader("Assistant Profile")

        prompt_profile_labels = {
            PromptProfile.BANKING_MODERNIZATION: "Banking Modernization",
            PromptProfile.IBM_I_APPLICATION_MODERNIZATION: "IBM i Application Modernization",
            PromptProfile.CLOUD_ARCHITECTURE: "Cloud Architecture",
            PromptProfile.GENERAL_ASSISTANT: "General Assistant",
        }
        st.selectbox(
            "Response Perspective",
            options=list(PromptProfile),
            key="prompt_profile",
            format_func=lambda profile: prompt_profile_labels[profile],
        )

        st.caption("Knowledge Library")
        st.caption(
            "Application readiness: "
            + ("ready" if readiness.ready else "not ready")
        )
        knowledge_summary = st.session_state.get(
            "knowledge_sync_summary"
        )
        if knowledge_summary:
            if knowledge_summary["failed"] == 0:
                st.write(
                    f"{len(st.session_state.startup_service.knowledge_manager.get_documents())} "
                    "documents synchronized"
                )
            else:
                st.warning(
                    f"Knowledge synchronization has "
                    f"{knowledge_summary['failed']} failed action(s)."
                )

        knowledge_refresh_message = st.session_state.pop(
            "knowledge_refresh_message",
            None,
        )
        if knowledge_refresh_message:
            if knowledge_refresh_message["type"] == "success":
                st.success(knowledge_refresh_message["text"])
            else:
                st.warning(knowledge_refresh_message["text"])

        if st.button(
            "Refresh Knowledge Library",
            use_container_width=True,
            disabled=("startup_service" not in st.session_state),
        ):
            with st.spinner("Refreshing knowledge library..."):
                try:
                    knowledge_sync_summary = (
                        st.session_state.startup_service
                        .refresh_knowledge_library()
                    )
                except Exception as error:
                    logger.error(
                        "Knowledge refresh failed (error_type=%s)",
                        type(error).__name__,
                    )
                    st.session_state.knowledge_refresh_message = {
                        "type": "warning",
                        "text": "Knowledge library refresh failed. Check the application logs and retry.",
                    }
                    st.rerun()
            st.session_state.knowledge_sync_summary = knowledge_sync_summary

            if knowledge_sync_summary["failed"] != 0:
                st.session_state.knowledge_refresh_message = {
                    "type": "warning",
                    "text": (
                        "Knowledge refresh completed with "
                        f"{knowledge_sync_summary['failed']} "
                        "failed action(s)."
                    ),
                }

            st.rerun()

        st.caption("Conversation")
        st.write(f"{len(st.session_state.messages)} messages")
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()


        st.divider()

        st.subheader("Documents")

        uploaded_file = st.file_uploader(
            "Upload PDF",
            type=["pdf"],
            key=f"document_upload_{st.session_state.document_uploader_key}",
        )

        if uploaded_file:
            document_hash = hashlib.sha256(uploaded_file.getvalue()).hexdigest()

            needs_indexing = document_hash != st.session_state.document_hash

            if needs_indexing:
                try:
                    with st.spinner("Indexing document..."):
                        if (
                            st.session_state.document_hash
                            and st.session_state.document_hash != document_hash
                        ):
                            pipeline.delete_document(
                                st.session_state.document_hash
                            )
                        result = pipeline.index_uploaded_document(
                            uploaded_file
                        )
                        if result.success:
                            st.session_state.document_indexed = True
                            st.session_state.document_name = uploaded_file.name
                            st.session_state.document_hash = document_hash
                            st.session_state.document_error = None
                        else:
                            st.session_state.document_indexed = False
                            st.session_state.document_error = result.error
                            st.session_state.document_name = None
                            st.session_state.document_hash = None
                except Exception as error:
                    logger.error(
                        "Uploaded-document indexing failed (error_type=%s)",
                        type(error).__name__,
                    )
                    st.error(
                        "The uploaded document could not be indexed. "
                        "Check the application logs and retry."
                    )
                    st.stop()
            else:
                logger.info("Document already indexed. Skipping indexing.")

            if st.session_state.document_indexed:
                if st.sidebar.button("Remove Document"):
                    try:
                        if st.session_state.document_hash:
                            pipeline.delete_document(
                                st.session_state.document_hash
                            )
                    except Exception as error:
                        logger.error(
                            "Uploaded-document removal failed (error_type=%s)",
                            type(error).__name__,
                        )
                        st.error(
                            "The uploaded document could not be removed. "
                            "Check the application logs and retry."
                        )
                    else:
                        st.session_state.document_indexed = False
                        st.session_state.document_name = None
                        st.session_state.document_hash = None
                        st.session_state.document_error = None
                        st.session_state.document_uploader_key += 1
                        st.rerun()
            elif st.session_state.document_error:
                st.sidebar.error(st.session_state.document_error)
        else:
            st.session_state.document_indexed = False
            st.session_state.document_name = None
            st.session_state.document_hash = None
            st.session_state.document_error = None


        st.divider()
        st.subheader("Answer Mode")
        available_modes = [
            RetrievalMode.GENERAL,
            RetrievalMode.LIBRARY,
        ]
        if st.session_state.document_indexed:
            available_modes.extend(
                [
                    RetrievalMode.UPLOADED,
                    RetrievalMode.HYBRID,
                ]
            )
        if st.session_state.retrieval_mode not in available_modes:
            st.session_state.retrieval_mode = RetrievalMode.GENERAL

        mode_labels = {
            RetrievalMode.GENERAL: "General Knowledge",
            RetrievalMode.LIBRARY: "Knowledge library",
            RetrievalMode.UPLOADED: "Uploaded document",
            RetrievalMode.HYBRID: "Library + Uploaded document",
        }
        st.radio(
            "Answer using",
            options = available_modes,
            key = "retrieval_mode",
            format_func = lambda mode:mode_labels[mode]
        )

        st.caption("Version")
        st.write(APP_VERSION)

    st.title(APP_NAME)
    st.caption(APP_DESCRIPTION)


    # Display previous messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

            source_references = message.get(
                "sources",
                [],
            )
            if source_references:
                source_labels = []
                for reference in source_references:
                    page = reference.get("page")
                    if page is None:
                        label = reference["source"]
                    else:
                        label = (
                            f"{reference['source']} "
                            f"(page {page})"
                        )
                    source_labels.append(label)

                st.caption(
                    "Sources: " + " \u00b7 ".join(source_labels)
                )

    # User input
    prompt = st.chat_input("Ask Bridge AI anything...")

    if prompt:

        # Add user message
        st.session_state.messages.append(
            {
                "role": "user",
                "content": prompt,
            }
        )

        with st.chat_message("user"):
            st.write(prompt)

        # Get LLM response with thinking status
        with st.status("Thinking...", expanded=False) as status:
            try:
                response, retrieval = chat_service.generate_response(
                    prompt=prompt,
                    messages=st.session_state.messages,
                    retrieval_mode=st.session_state.retrieval_mode,
                    document_hash=st.session_state.document_hash,
                    prompt_profile=st.session_state.prompt_profile,
                    provider_name=st.session_state.provider_name,
                    model=st.session_state.model_name,
                )

                status.update(
                    label="Completed",
                    state="complete",
                )
            except Exception as e:
                if not isinstance(e, ChatProviderError):
                    logger.error(
                        "Chat operation failed (error_type=%s)",
                        type(e).__name__,
                    )
                status.update(
                    label="Error",
                    state="error",
                )
                st.error(
                    "Bridge AI could not complete this request. "
                    "Check the application logs and retry."
                )
                st.stop()

        source_references = build_source_references(
            retrieval
        )
        # Add assistant message
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": response,
                "sources": source_references,
            }
        )

        with st.chat_message("assistant"):
            st.write(response)

            if source_references:
                source_labels = []
                for reference in source_references:
                    page = reference.get("page")
                    if page is None:
                        label = reference["source"]
                    else:
                        label = (
                            f"{reference['source']} "
                            f"(page {page})"
                        )
                    source_labels.append(label)
                st.caption(
                    "Sources: " + " \u00b7 ".join(source_labels)
                )
