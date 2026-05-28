"""Unit tests for IGClient.ig_service property (REQ-14).

Verifies that the ig_service property exposes the underlying IGService
instance without triggering a new authentication call.
"""

import types
from unittest.mock import MagicMock, patch, call

import pytest

from ig_client import IGClient

# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #


def _make_ig_client() -> IGClient:
    """Return an IGClient with all network calls mocked out.

    Patches IGService at the point of import inside ig_client so that no
    real authentication happens during construction.
    """
    mock_svc = MagicMock()
    mock_svc.create_session.return_value = None

    with patch("ig_client.IGService", return_value=mock_svc), patch.dict(
        "os.environ",
        {
            "ig_username": "test_user",
            "ig_password": "test_pass",
            "ig_api_key": "test_key",
            "ig_acc_number": "ACC123",
            "ig_acc_type": "DEMO",
        },
    ):
        client = IGClient()

    # Stash the mock so tests can assert against it
    client._test_mock_svc = mock_svc
    return client


# --------------------------------------------------------------------------- #
# IGClient.ig_service — REQ-14                                                 #
# --------------------------------------------------------------------------- #


class TestIGClientIGServiceProperty:
    """Tests for the ig_service property on IGClient (REQ-14)."""

    def test_ig_service_property_returns_underlying_svc(self):
        """REQ-14 scenario 1: ig_service returns the active IGService object."""
        client = _make_ig_client()

        result = client.ig_service

        assert result is client._svc

    def test_ig_service_property_does_not_re_authenticate(self):
        """REQ-14: accessing ig_service must not trigger create_session again."""
        client = _make_ig_client()
        initial_call_count = client._test_mock_svc.create_session.call_count

        _ = client.ig_service

        # create_session should not have been called an additional time
        assert client._test_mock_svc.create_session.call_count == initial_call_count

    def test_ig_service_returns_same_instance_on_multiple_accesses(self):
        """ig_service must return the same object reference on repeated access."""
        client = _make_ig_client()

        first = client.ig_service
        second = client.ig_service

        assert first is second
