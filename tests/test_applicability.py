from datetime import date

from veltrix_assistant.ingestion.schemas import (
    AccessLevel,
    DocumentMetadata,
    DocumentType,
)
from veltrix_assistant.policy.applicability import (
    DocumentAudience,
    PolicyEvidence,
    document_precedence_key,
    get_applicable_documents,
    get_upcoming_winners,
    is_document_applicable,
    is_document_readable,
    resolve_policy_evidence,
    select_precedence_winners,
)


def _document(
    doc_id: str,
    *,
    doc_type: DocumentType = "policy",
    regions: tuple[str, ...] = ("GLOBAL",),
    effective_date: date | str = "2025-01-01",
    access_level: AccessLevel = "all",
) -> DocumentMetadata:
    return DocumentMetadata(
        doc_id=doc_id,
        file=f"{doc_id}.md",
        title=doc_id,
        doc_type=doc_type,
        regions=regions,
        effective_date=(
            date.fromisoformat(effective_date)
            if isinstance(effective_date, str)
            else effective_date
        ),
        access_level=access_level,
    )


def test_eligibility_uses_role_region_and_reference_date() -> None:
    audience = DocumentAudience(region="IN", role="employee")
    documents = [
        _document("GLOBAL"),
        _document("INDIA", regions=("IN",)),
        _document("US", regions=("US",)),
        _document("MANAGER", access_level="managers"),
        _document("HR", access_level="hr"),
        _document("FUTURE", effective_date="2027-01-01"),
    ]

    assert {doc.doc_id for doc in get_applicable_documents(
        documents, audience, date(2026, 12, 31)
    )} == {"GLOBAL", "INDIA"}
    assert not is_document_applicable(
        documents[5], audience, date(2026, 12, 31)
    )
    assert is_document_readable(documents[5], audience)


def test_authority_precedes_region_scope_and_recency() -> None:
    audience = DocumentAudience(region="DE", role="employee")
    faq = _document(
        "FAQ-NEW",
        doc_type="faq",
        regions=("DE",),
        effective_date="2026-12-01",
    )
    global_policy = _document(
        "POLICY-OLD",
        doc_type="policy",
        effective_date="2025-01-01",
    )

    assert document_precedence_key(faq, audience) < document_precedence_key(
        global_policy, audience
    )
    assert select_precedence_winners([faq, global_policy], audience) == (
        global_policy,
    )


def test_region_scope_precedes_recency() -> None:
    audience = DocumentAudience(region="DE", role="employee")
    regional = _document(
        "DE-ADDENDUM",
        doc_type="addendum",
        regions=("DE",),
        effective_date="2025-01-01",
    )
    newer_global = _document(
        "GLOBAL-NEW",
        doc_type="policy",
        effective_date="2026-12-01",
    )

    assert document_precedence_key(regional, audience) > document_precedence_key(
        newer_global, audience
    )
    assert select_precedence_winners([regional, newer_global], audience) == (
        regional,
    )


def test_recency_breaks_ties_after_authority_and_scope() -> None:
    audience = DocumentAudience(region="IN", role="employee")
    older = _document("V1", effective_date="2025-01-01")
    newer = _document("V2", effective_date="2026-04-01")

    assert select_precedence_winners([older, newer], audience) == (newer,)


def test_exact_precedence_tie_retains_every_winner() -> None:
    audience = DocumentAudience(region="IN", role="employee")
    first = _document("POLICY-A", effective_date="2026-04-01")
    second = _document("POLICY-B", effective_date="2026-04-01")

    assert select_precedence_winners([first, second], audience) == (first, second)


def test_policy_resolution_reports_conflicting_values_for_a_true_tie() -> None:
    audience = DocumentAudience(region="IN", role="employee")
    first = _document("POLICY-A", effective_date="2026-04-01")
    second = _document("POLICY-B", effective_date="2026-04-01")
    older = _document("POLICY-OLD", effective_date="2025-01-01")
    evidence = [
        PolicyEvidence(
            rule_key="annual_leave_days",
            value="20 days",
            document=first,
            locator="Section 2",
        ),
        PolicyEvidence(
            rule_key="annual_leave_days",
            value="22 days",
            document=second,
            locator="Section 3",
        ),
        PolicyEvidence(
            rule_key="annual_leave_days",
            value="18 days",
            document=older,
            locator="Section 1",
        ),
        PolicyEvidence(
            rule_key="sick_leave_days",
            value="10 days",
            document=first,
            locator="Section 4",
        ),
    ]

    resolutions = resolve_policy_evidence(
        evidence,
        audience,
        date(2026, 12, 31),
    )

    assert len(resolutions) == 2
    annual_leave, sick_leave = resolutions
    assert annual_leave.rule_key == "annual_leave_days"
    assert annual_leave.values == ("20 days", "22 days")
    assert annual_leave.unresolved_conflict
    assert {item.document.doc_id for item in annual_leave.evidence} == {
        "POLICY-A",
        "POLICY-B",
    }
    assert {item.locator for item in annual_leave.evidence} == {
        "Section 2",
        "Section 3",
    }
    assert sick_leave.values == ("10 days",)
    assert not sick_leave.unresolved_conflict


def test_upcoming_winner_is_returned_with_a_regional_precedence_check() -> None:
    audience = DocumentAudience(region="IN", role="employee")
    current_global = _document("GLOBAL-V2", effective_date="2026-04-01")
    future_global = _document("GLOBAL-V3", effective_date="2027-01-01")
    future_faq = _document(
        "FUTURE-FAQ",
        doc_type="faq",
        effective_date="2026-12-01",
    )
    current_regional = _document(
        "INDIA-ADDENDUM",
        doc_type="addendum",
        regions=("IN",),
        effective_date="2025-01-01",
    )

    upcoming = get_upcoming_winners(
        [current_global, future_global, future_faq],
        audience,
        date(2026, 11, 1),
    )
    assert upcoming == (future_global,)

    upcoming_with_regional = get_upcoming_winners(
        [current_regional, future_global],
        audience,
        date(2026, 11, 1),
    )
    assert upcoming_with_regional == ()


def test_upcoming_does_not_expose_inaccessible_future_docs() -> None:
    audience = DocumentAudience(region="IN", role="employee")
    current = _document("CURRENT")
    restricted = _document(
        "RESTRICTED-FUTURE",
        effective_date="2026-12-01",
        access_level="hr",
    )

    assert get_upcoming_winners(
        [current, restricted], audience, date(2026, 11, 1)
    ) == ()
