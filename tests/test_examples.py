"""
Integration tests that run example scripts and verify outputs.
"""

import os
import sys
import subprocess
import pytest
import tempfile

# Add the package to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Timeout in seconds (10 minutes)
TIMEOUT_SECONDS = 600


def run_example_script(script_name: str, working_dir: str) -> int:
    """Run an example script and return the exit code."""
    script_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "xgboost_interp", "examples", script_name
    )
    
    result = subprocess.run(
        [sys.executable, script_path],
        cwd=working_dir,
        capture_output=True,
        text=True,
        timeout=TIMEOUT_SECONDS
    )
    
    if result.returncode != 0:
        print(f"STDOUT:\n{result.stdout}")
        print(f"STDERR:\n{result.stderr}")
    
    return result.returncode


def verify_output_files(output_dir: str, min_png_count: int = 1) -> None:
    """
    Verify that output PNG files exist.
    
    Args:
        output_dir: Path to the output directory
        min_png_count: Minimum number of PNG files expected
    
    Raises:
        AssertionError: If validation fails
    """
    # Check output directory exists
    assert os.path.exists(output_dir), f"Output directory not found: {output_dir}"
    
    # Get all PNG files
    png_files = [f for f in os.listdir(output_dir) if f.endswith('.png')]
    assert len(png_files) >= min_png_count, (
        f"Expected at least {min_png_count} PNG files, found {len(png_files)}"
    )
    
    print(f"✓ Verified {len(png_files)} PNG files in {output_dir}")


def test_iris_example():
    """Test that the Iris classification example runs successfully."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Run the example
        exit_code = run_example_script("iris_classification_example.py", tmpdir)
        
        assert exit_code == 0, "Iris example failed to run"
        
        # Verify output PNG files exist and are fresh
        output_dir = os.path.join(tmpdir, "examples", "iris", "output")
        verify_output_files(output_dir, min_png_count=1)


def test_california_housing_example():
    """Test that the California Housing example runs successfully."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Run the example
        exit_code = run_example_script("california_housing_example.py", tmpdir)
        
        assert exit_code == 0, "California Housing example failed to run"
        
        # Verify output PNG files exist and are fresh
        output_dir = os.path.join(tmpdir, "examples", "california_housing", "output")
        verify_output_files(output_dir, min_png_count=1)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

