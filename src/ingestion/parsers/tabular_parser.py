"""Parser for tabular data (CSV, Excel) files."""

import logging
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
from pathlib import Path
import json
from dataclasses import dataclass
import numpy as np

from ...core.error_handling import error_handler, DocumentProcessingError

logger = logging.getLogger(__name__)

@dataclass
class TabularDocument:
    """Result of tabular document parsing."""
    dataframes: Dict[str, pd.DataFrame]
    metadata: Dict[str, Any]
    representations: List[Dict[str, Any]]

class TabularParser:
    """
    Parser for CSV files only.
    
    Note: Excel files (.xlsx, .xls) are handled by Docling parser
    which provides better structure extraction and table understanding.
    
    This parser focuses on CSV files which Docling doesn't support.
    """
    
    def parse(self, file_path: str) -> TabularDocument:
        """Parse a CSV file."""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise DocumentProcessingError(
                f"File not found: {file_path}",
                "tabular_parser",
                {"file_path": str(file_path)}
            )
        
        file_extension = file_path.suffix.lower()
        
        # Only handle CSV
        if file_extension == '.csv':
            return self._parse_csv(file_path)
        else:
            raise DocumentProcessingError(
                f"TabularParser only handles CSV files. Use Docling for Excel.",
                "tabular_parser",
                {"file_path": str(file_path), "extension": file_extension}
            )
    
    def _parse_csv(self, file_path: Path) -> TabularDocument:
        """Parse CSV file."""
        try:
            # Read CSV with proper encoding detection
            df = pd.read_csv(file_path, encoding='utf-8')
            
            # Handle encoding issues
            if df.empty:
                df = pd.read_csv(file_path, encoding='latin-1')
            
            dataframes = {"Sheet1": df}
            metadata = self._generate_metadata(df, "Sheet1")
            representations = self._generate_representations(df, "Sheet1")
            
            return TabularDocument(
                dataframes=dataframes,
                metadata=metadata,
                representations=representations
            )
            
        except Exception as e:
            raise DocumentProcessingError(
                f"Failed to parse CSV file: {e}",
                "tabular_parser",
                {"file_path": str(file_path)}
            )
    
    
    def _generate_metadata(self, df: pd.DataFrame, sheet_name: str) -> Dict[str, Any]:
        """Generate metadata for a dataframe."""
        metadata = {
            "sheet_name": sheet_name,
            "rows": len(df),
            "columns": len(df.columns),
            "column_names": list(df.columns),
            "column_types": {col: str(df[col].dtype) for col in df.columns},
            "memory_usage": df.memory_usage(deep=True).sum(),
            "has_nulls": df.isnull().any().any(),
            "null_counts": {col: int(df[col].isnull().sum()) for col in df.columns},
            "file_type": "tabular"
        }
        
        # Add statistical metadata for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            metadata["numeric_statistics"] = {}
            for col in numeric_cols:
                stats = {
                    "mean": float(df[col].mean()) if not pd.isna(df[col].mean()) else None,
                    "std": float(df[col].std()) if not pd.isna(df[col].std()) else None,
                    "min": float(df[col].min()) if not pd.isna(df[col].min()) else None,
                    "max": float(df[col].max()) if not pd.isna(df[col].max()) else None,
                    "median": float(df[col].median()) if not pd.isna(df[col].median()) else None,
                    "quartiles": {
                        "25%": float(df[col].quantile(0.25)) if not pd.isna(df[col].quantile(0.25)) else None,
                        "75%": float(df[col].quantile(0.75)) if not pd.isna(df[col].quantile(0.75)) else None
                    }
                }
                metadata["numeric_statistics"][col] = stats
        
        return metadata
    
    def _generate_representations(self, df: pd.DataFrame, sheet_name: str) -> List[Dict[str, Any]]:
        """Generate multiple text representations of the dataframe."""
        representations = []
        
        # 1. Column summary representation
        column_summary = self._generate_column_summary(df, sheet_name)
        representations.append({
            "type": "column_summary",
            "content": column_summary,
            "sheet_name": sheet_name,
            "description": f"Column schema and data type summary for {sheet_name}"
        })
        
        # 2. Statistical summary representation
        if len(df.select_dtypes(include=[np.number]).columns) > 0:
            stats_summary = self._generate_statistical_summary(df, sheet_name)
            representations.append({
                "type": "statistical_summary",
                "content": stats_summary,
                "sheet_name": sheet_name,
                "description": f"Statistical summary for numeric columns in {sheet_name}"
            })
        
        # 3. Sample rows representation
        sample_rows = self._generate_sample_rows(df, sheet_name)
        representations.append({
            "type": "sample_rows",
            "content": sample_rows,
            "sheet_name": sheet_name,
            "description": f"Sample data rows from {sheet_name}"
        })
        
        # 4. Data quality summary
        quality_summary = self._generate_quality_summary(df, sheet_name)
        representations.append({
            "type": "quality_summary",
            "content": quality_summary,
            "sheet_name": sheet_name,
            "description": f"Data quality assessment for {sheet_name}"
        })
        
        return representations
    
    def _generate_column_summary(self, df: pd.DataFrame, sheet_name: str) -> str:
        """Generate column schema summary."""
        summary = f"Column Summary for Sheet: {sheet_name}\n"
        summary += "=" * 50 + "\n\n"
        
        summary += f"Total Columns: {len(df.columns)}\n"
        summary += f"Total Rows: {len(df)}\n\n"
        
        summary += "Column Details:\n"
        for i, col in enumerate(df.columns, 1):
            dtype = str(df[col].dtype)
            non_null_count = df[col].count()
            null_count = df[col].isnull().sum()
            null_percentage = (null_count / len(df)) * 100
            
            summary += f"{i}. {col}\n"
            summary += f"   Type: {dtype}\n"
            summary += f"   Non-null values: {non_null_count}\n"
            summary += f"   Null values: {null_count} ({null_percentage:.1f}%)\n"
            
            # Add sample values for non-numeric columns
            if df[col].dtype == 'object' and non_null_count > 0:
                sample_values = df[col].dropna().head(3).tolist()
                summary += f"   Sample values: {sample_values}\n"
            
            summary += "\n"
        
        return summary
    
    def _generate_statistical_summary(self, df: pd.DataFrame, sheet_name: str) -> str:
        """Generate statistical summary for numeric columns."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        summary = f"Statistical Summary for Sheet: {sheet_name}\n"
        summary += "=" * 50 + "\n\n"
        
        summary += f"Numeric Columns: {len(numeric_cols)}\n\n"
        
        for col in numeric_cols:
            summary += f"Column: {col}\n"
            summary += f"  Mean: {df[col].mean():.2f}\n"
            summary += f"  Median: {df[col].median():.2f}\n"
            summary += f"  Std Dev: {df[col].std():.2f}\n"
            summary += f"  Min: {df[col].min():.2f}\n"
            summary += f"  Max: {df[col].max():.2f}\n"
            summary += f"  25th Percentile: {df[col].quantile(0.25):.2f}\n"
            summary += f"  75th Percentile: {df[col].quantile(0.75):.2f}\n"
            summary += "\n"
        
        return summary
    
    def _generate_sample_rows(self, df: pd.DataFrame, sheet_name: str) -> str:
        """Generate sample rows representation."""
        sample_size = min(self.max_rows_for_sample, len(df))
        sample_df = df.head(sample_size)
        
        summary = f"Sample Data for Sheet: {sheet_name}\n"
        summary += "=" * 50 + "\n\n"
        
        summary += f"Showing first {sample_size} rows out of {len(df)} total rows:\n\n"
        
        # Convert to string representation
        summary += sample_df.to_string(index=False)
        
        return summary
    
    def _generate_quality_summary(self, df: pd.DataFrame, sheet_name: str) -> str:
        """Generate data quality assessment."""
        summary = f"Data Quality Summary for Sheet: {sheet_name}\n"
        summary += "=" * 50 + "\n\n"
        
        # Overall quality metrics
        total_cells = df.size
        non_null_cells = df.count().sum()
        null_cells = total_cells - non_null_cells
        completeness = (non_null_cells / total_cells) * 100
        
        summary += f"Overall Completeness: {completeness:.1f}%\n"
        summary += f"Total Cells: {total_cells:,}\n"
        summary += f"Non-null Cells: {non_null_cells:,}\n"
        summary += f"Null Cells: {null_cells:,}\n\n"
        
        # Column-wise quality
        summary += "Column-wise Quality:\n"
        for col in df.columns:
            col_completeness = (df[col].count() / len(df)) * 100
            summary += f"  {col}: {col_completeness:.1f}% complete\n"
        
        # Data type distribution
        dtype_counts = df.dtypes.value_counts()
        summary += f"\nData Types:\n"
        for dtype, count in dtype_counts.items():
            summary += f"  {dtype}: {count} columns\n"
        
        return summary

# Global parser instance
tabular_parser = TabularParser()