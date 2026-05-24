from typing import Annotated, Optional, List, Dict, Any
import os
import pandas as pd

from autogen.agentchat.group import ContextVariables, ReplyResult

# Define SCRIPT_DIR somewhere central in your project
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def Profiling_Data(
    csv_file_path: Optional[str] = None,
    sample_rows: int = 5
) -> Dict[str, Any]:
    """
    Discovers and profiles CSV files in the script directory.
    
    This function:
    1. If csv_file_path is None, scans SCRIPT_DIR for CSV files and selects the first one
    2. If csv_file_path is provided, uses that file directly
    3. Extracts metadata including columns, types, and sample data
    4. Returns a dictionary containing all metadata
    
    Parameters
    ----------
    csv_file_path : str, optional
        Path to the CSV file to profile. If None, auto-discovers CSV in SCRIPT_DIR.
    sample_rows : int, default=5
        Number of rows to sample for type inference and preview.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - file_info: Dict with filename, filepath, size info
        - columns: List of column names
        - column_count: int
        - row_count: int
        - column_details: Dict mapping column names to their details
        - numerical_columns: List of numerical column info
        - categorical_columns: List of categorical column info
        - sample_data: List of dictionaries representing sample rows
        - summary: Human-readable summary string
        - error: Error message if something went wrong (only present on error)
    """
    
    base_dir = SCRIPT_DIR
    
    # -----------------------
    # Step 1: Discover or validate CSV file
    # -----------------------
    if csv_file_path is None:
        try:
            if not os.path.isdir(base_dir):
                return {"error": f"Directory not found: {base_dir}"}

            files = [
                f for f in os.listdir(base_dir)
                if f.lower().endswith(".csv")
            ]

            if not files:
                return {"error": f"No CSV files found in {base_dir}"}

            # Select the first CSV file to profile
            csv_filename = files[0]
            csv_file_path = os.path.join(base_dir, csv_filename)

        except Exception as e:
            return {"error": f"Error scanning directory '{base_dir}': {e}"}
    
    # Final existence check
    if not os.path.exists(csv_file_path):
        return {"error": f"CSV file not found: {csv_file_path}"}

    # -----------------------
    # Step 2: Profile the CSV file
    # -----------------------
    
    # Initialize metadata container
    metadata: Dict[str, Any] = {
        "file_info": {},
        "columns": [],
        "column_count": 0,
        "row_count": 0,
        "column_details": {},
        "numerical_columns": [],
        "categorical_columns": [],
        "sample_data": [],
        "summary": "",
    }

    try:
        # File info
        file_size = os.path.getsize(csv_file_path)
        metadata["file_info"] = {
            "filename": os.path.basename(csv_file_path),
            "filepath": os.path.abspath(csv_file_path),
            "size_bytes": file_size,
            "size_readable": (
                f"{file_size / 1024:.2f} KB"
                if file_size < 1024 * 1024
                else f"{file_size / (1024 * 1024):.2f} MB"
            ),
        }

        # Row count (excluding header)
        with open(csv_file_path, "r", encoding="utf-8") as f:
            row_count = sum(1 for _ in f) - 1
        if row_count < 0:
            return {"error": f"CSV file is empty: {csv_file_path}"}

        metadata["row_count"] = row_count

        # Sample rows for header + type inference
        df_sample = pd.read_csv(csv_file_path, nrows=sample_rows)

        metadata["columns"] = df_sample.columns.tolist()
        metadata["column_count"] = len(df_sample.columns)

        # Column typing and categorization
        for col_idx, col_name in enumerate(df_sample.columns):
            series = df_sample[col_name]
            dtype = str(series.dtype)

            if pd.api.types.is_numeric_dtype(series):
                category = "numerical"
                metadata["numerical_columns"].append(
                    {
                        "column_index": col_idx,
                        "column_name": col_name,
                        "dtype": dtype,
                    }
                )
            else:
                category = "categorical"
                metadata["categorical_columns"].append(
                    {
                        "column_index": col_idx,
                        "column_name": col_name,
                        "dtype": dtype,
                    }
                )

            metadata["column_details"][col_name] = {
                "column_index": col_idx,
                "dtype": dtype,
                "category": category,
            }

        # Sample data (for downstream agents' context)
        metadata["sample_data"] = df_sample.to_dict("records")

        # Build summary text
        summary_parts: List[str] = [
            f"CSV File: {metadata['file_info']['filename']}",
            f"File Size: {metadata['file_info']['size_readable']}",
            f"Dimensions: {metadata['row_count']} rows × {metadata['column_count']} columns",
            f"\n=== NUMERICAL COLUMNS ({len(metadata['numerical_columns'])}) ===",
        ]

        if metadata["numerical_columns"]:
            for col in metadata["numerical_columns"]:
                summary_parts.append(
                    f"  Column {col['column_index']}: "
                    f"'{col['column_name']}' (Type: {col['dtype']})"
                )
        else:
            summary_parts.append("  (None)")

        summary_parts.append(
            f"\n=== CATEGORICAL COLUMNS ({len(metadata['categorical_columns'])}) ==="
        )

        if metadata["categorical_columns"]:
            for col in metadata["categorical_columns"]:
                summary_parts.append(
                    f"  Column {col['column_index']}: "
                    f"'{col['column_name']}' (Type: {col['dtype']})"
                )
        else:
            summary_parts.append("  (None)")

        summary_parts.append(
            f"\nNote: Types/categories inferred from first {sample_rows} rows only."
        )
        summary_parts.append(
            "Full dataset NOT loaded to stay efficient for large files."
        )
        summary_parts.append(
            "Column indices are 0-based (first column = 0, second = 1, etc.)."
        )

        metadata["summary"] = "\n".join(summary_parts)

        return metadata

    except pd.errors.EmptyDataError:
        return {"error": f"CSV file is empty or malformed: {csv_file_path}"}
    except pd.errors.ParserError as e:
        return {"error": f"Error parsing CSV file: {e}"}
    except Exception as e:
        return {"error": f"Unexpected error processing CSV file: {e}"}


def Context_Update(
    metadata: Dict[str, Any],
    context_variables: Optional[ContextVariables] = None,
) -> ReplyResult:
    """
    Updates context variables with the provided metadata.
    
    This function takes metadata (typically from Profiling_Data) and stores it
    in the shared context_variables under the 'Metadata' key.
    
    Parameters
    ----------
    metadata : Dict[str, Any]
        Metadata dictionary to store in context. Should contain keys like
        'file_info', 'columns', 'summary', etc.
    context_variables : ContextVariables, optional
        Injected by AG2. If None, a fresh instance is created.

    Returns
    -------
    ReplyResult
        message: str
            Human-readable description of what was done.
        context_variables: ContextVariables
            Updated with metadata under 'Metadata' key.
    """
    
    # Ensure we always have a ContextVariables instance
    if context_variables is None:
        context_variables = ContextVariables()
    
    # Check if metadata contains an error
    if "error" in metadata:
        context_variables["MetadataError"] = metadata["error"]
        return ReplyResult(
            message=metadata["error"],
            context_variables=context_variables
        )
    
    # Store metadata in shared context
    context_variables["Metadata"] = metadata
    
    # Build success message
    msg = f"✓ Metadata extraction successful!\n\n{metadata.get('summary', 'Metadata stored.')}"
    
    return ReplyResult(message=msg, context_variables=context_variables)


def Get_Dataset_Metadata(
    context_variables: Annotated[Optional[ContextVariables], "Injected by AG2, contains shared state"] = None,
) -> ReplyResult:
    """
    Retrieve the dataset metadata previously stored under 'Metadata' in ContextVariables.
    
    THIS FUNCTION TAKES NO ARGUMENTS (except context_variables which is auto-injected).
    Simply call it and it will return the stored metadata.

    Expected usage:
    - Called after the Profile tool has run successfully.
    - Reads from context_variables["Metadata"].
    
    Behavior:
    - If Metadata exists:
        - Returns a ReplyResult with a concise human-readable summary plus
          a JSON-style view of the metadata for downstream agents.
    - If Metadata is missing:
        - Looks for 'MetadataError' and returns that if present.
        - Otherwise returns a helpful message explaining that Profile must be run first.

    Parameters
    ----------
    context_variables : ContextVariables, optional
        Auto-injected by AG2. Contains shared state including 'Metadata' key.

    Returns
    -------
    ReplyResult
        message: str
            Explanation + (if available) the metadata content.
        context_variables: ContextVariables
            Passed through unchanged.
    """

    # Ensure we always have a ContextVariables instance
    if context_variables is None:
        context_variables = ContextVariables()
        return ReplyResult(
            message=(
                "No shared context was provided, so no dataset metadata is available. "
                "Please run the Profile tool first in a shared group context."
            ),
            context_variables=context_variables,
        )

    # Try to retrieve stored metadata
    # AG2 nested chats may wrap context under a "data" key, so check both locations
    metadata = None
    try:
        # Try direct access first
        metadata = context_variables["Metadata"]
    except (KeyError, TypeError):
        try:
            # Try under "data" key (for nested chats with register_function)
            if "data" in context_variables and isinstance(context_variables["data"], dict):
                metadata = context_variables["data"].get("Metadata")
        except (AttributeError, TypeError, KeyError):
            metadata = None

    if metadata is None:
        # If there was an error recorded earlier, surface it
        metadata_error = context_variables.get("MetadataError", None)
        if metadata_error:
            return ReplyResult(
                message=(
                    "No valid dataset metadata found in context.\n\n"
                    f"Last recorded metadata error:\n{metadata_error}"
                ),
                context_variables=context_variables,
            )

        # Generic fallback
        return ReplyResult(
            message=(
                "No dataset metadata found in context under 'Metadata'. "
                "Run the Profile tool on a CSV file first to populate it."
            ),
            context_variables=context_variables,
        )

    # Build a concise human-readable view
    file_info = metadata.get("file_info", {})
    rows = metadata.get("row_count", "unknown")
    cols = metadata.get("column_count", "unknown")
    num_cols = len(metadata.get("numerical_columns", []))
    cat_cols = len(metadata.get("categorical_columns", []))
    summary_text = metadata.get("summary", "")

    msg_lines = [
        "✅ Retrieved dataset metadata from context (key: 'Metadata').",
        "",
        f"File: {file_info.get('filename', 'unknown')}",
        f"Path: {file_info.get('filepath', 'unknown')}",
        f"Size: {file_info.get('size_readable', 'unknown')}",
        f"Shape: {rows} rows × {cols} columns",
        f"Numerical columns: {num_cols}",
        f"Categorical columns: {cat_cols}",
        "",
        "Summary:",
        summary_text if summary_text else "(No detailed summary stored.)",
        "",
        "Raw metadata object is available in context_variables['Metadata'] "
        "for downstream tools/agents."
    ]

    return ReplyResult(
        message="\n".join(msg_lines),
        context_variables=context_variables,
    )

