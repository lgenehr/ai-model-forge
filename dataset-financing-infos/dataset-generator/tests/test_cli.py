"""Tests for CLI module."""

import pytest
from typer.testing import CliRunner

from src.main import app

runner = CliRunner()


class TestCLI:
    """Tests for CLI commands."""

    def test_help(self):
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "dataset-generator" in result.output.lower() or "usage" in result.output.lower()

    def test_collect_help(self):
        result = runner.invoke(app, ["collect", "--help"])
        assert result.exit_code == 0
        assert "sources" in result.output.lower()
        assert "topics" in result.output.lower()

    def test_process_help(self):
        result = runner.invoke(app, ["process", "--help"])
        assert result.exit_code == 0
        assert "input" in result.output.lower()
        assert "quality" in result.output.lower()

    def test_format_help(self):
        result = runner.invoke(app, ["format", "--help"])
        assert result.exit_code == 0
        assert "format" in result.output.lower()

    def test_stats_help(self):
        result = runner.invoke(app, ["stats", "--help"])
        assert result.exit_code == 0
        assert "input" in result.output.lower()

    def test_validate_help(self):
        result = runner.invoke(app, ["validate", "--help"])
        assert result.exit_code == 0
        assert "sample" in result.output.lower()


class TestCLIIntegration:
    """Integration tests for CLI (may require network access)."""

    @pytest.mark.slow
    def test_collect_dry_run(self, tmp_path):
        # This would need actual collection - skip for now
        pass

    def test_stats_empty_dir(self, tmp_path):
        result = runner.invoke(app, ["stats", "--input-dir", str(tmp_path)])
        # Should handle empty directory gracefully
        assert result.exit_code == 0

    def test_validate_empty_dir(self, tmp_path):
        result = runner.invoke(app, ["validate", "--input-dir", str(tmp_path)])
        # Should handle empty directory
        assert result.exit_code == 0
