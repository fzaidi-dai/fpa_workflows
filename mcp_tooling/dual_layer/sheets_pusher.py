"""
Smart Google Sheets Pusher for Dual-Layer Execution

Intelligently pushes formulas, values, and creates native Google Sheets elements
based on context and content type.
"""

from typing import Any, Dict, List, Optional, Union
from enum import Enum
import logging

from .data_models import CellContext, CellType

# Set up logger
logger = logging.getLogger(__name__)


class PushStrategy(Enum):
    """Different strategies for pushing data to sheets"""
    VALUES_ONLY = "values_only"
    FORMULA_ONLY = "formula_only"
    BOTH_WITH_NOTE = "both_with_note"
    PIVOT_TABLE = "pivot_table"
    CHART = "chart"


class SmartSheetsPusher:
    """Intelligently pushes formulas and values to Google Sheets"""
    
    def __init__(self, batch_size: int = 50, enable_notes: bool = True):
        """
        Initialize the smart sheets pusher
        
        Args:
            batch_size: Maximum number of operations to batch together
            enable_notes: Whether to add validation notes to cells
        """
        self.batch_size = batch_size
        self.enable_notes = enable_notes
        self.pending_operations: List[Dict[str, Any]] = []
        self.operation_count = 0
    
    def push_with_context(self, cell_context: CellContext) -> bool:
        """
        Push data based on cell context
        
        Args:
            cell_context: Context information for the cell
            
        Returns:
            True if successful, False otherwise
        """
        try:
            strategy = self._determine_strategy(cell_context)
            
            if strategy == PushStrategy.VALUES_ONLY:
                return self._push_values_only(cell_context)
            elif strategy == PushStrategy.FORMULA_ONLY:
                return self._push_formula_only(cell_context)
            elif strategy == PushStrategy.BOTH_WITH_NOTE:
                return self._push_both_with_note(cell_context)
            elif strategy == PushStrategy.PIVOT_TABLE:
                return self._push_pivot_table(cell_context)
            elif strategy == PushStrategy.CHART:
                return self._push_chart(cell_context)
            
            return False
            
        except Exception as e:
            logger.error(f"Error pushing context {cell_context.range_spec}: {e}")
            return False
    
    def _determine_strategy(self, cell_context: CellContext) -> PushStrategy:
        """Determine the best push strategy for the given context"""
        
        if cell_context.cell_type == CellType.SOURCE_DATA:
            return PushStrategy.VALUES_ONLY
            
        elif cell_context.cell_type == CellType.CALCULATION:
            # Simple calculations use formula only
            return PushStrategy.FORMULA_ONLY
            
        elif cell_context.cell_type == CellType.COMPLEX_CALCULATION:
            # Complex calculations get formula with validation note
            return PushStrategy.BOTH_WITH_NOTE
            
        elif cell_context.cell_type == CellType.PIVOT_TABLE:
            return PushStrategy.PIVOT_TABLE
            
        elif cell_context.cell_type == CellType.CHART_DATA:
            return PushStrategy.CHART
        
        # Default fallback
        return PushStrategy.FORMULA_ONLY if cell_context.formula else PushStrategy.VALUES_ONLY
    
    def _push_values_only(self, cell_context: CellContext) -> bool:
        """Push raw values only - for source data"""
        if not cell_context.values:
            logger.warning(f"No values to push for {cell_context.range_spec}")
            return False
        
        operation = {
            "type": "update_values",
            "range": cell_context.range_spec,
            "values": cell_context.values,
            "value_input_option": "RAW"
        }
        
        self.pending_operations.append(operation)
        logger.info(f"Queued values push to {cell_context.range_spec}")
        
        return self._maybe_execute_batch()
    
    def _push_formula_only(self, cell_context: CellContext) -> bool:
        """Push formula only - let Sheets calculate"""
        if not cell_context.formula:
            logger.warning(f"No formula to push for {cell_context.range_spec}")
            return False
        
        operation = {
            "type": "update_values", 
            "range": cell_context.range_spec,
            "values": [[cell_context.formula]],
            "value_input_option": "USER_ENTERED"
        }
        
        self.pending_operations.append(operation)
        logger.info(f"Queued formula push to {cell_context.range_spec}: {cell_context.formula}")
        
        return self._maybe_execute_batch()
    
    def _push_both_with_note(self, cell_context: CellContext) -> bool:
        """Push formula with validation note for complex calculations"""
        
        # First push the formula
        if cell_context.formula:
            formula_operation = {
                "type": "update_values",
                "range": cell_context.range_spec, 
                "values": [[cell_context.formula]],
                "value_input_option": "USER_ENTERED"
            }
            self.pending_operations.append(formula_operation)
        
        # Add validation note if enabled
        if self.enable_notes and cell_context.computed_value is not None:
            note_text = f"Polars computed: {cell_context.computed_value}"
            if cell_context.validation_note:
                note_text += f"\\n{cell_context.validation_note}"
            
            note_operation = {
                "type": "add_note",
                "range": cell_context.range_spec,
                "note": note_text
            }
            self.pending_operations.append(note_operation)
        
        logger.info(f"Queued complex formula push to {cell_context.range_spec}")
        
        return self._maybe_execute_batch()
    
    def _push_pivot_table(self, cell_context: CellContext) -> bool:
        """Create native Google Sheets pivot table"""
        if not cell_context.pivot_config:
            logger.warning(f"No pivot config for {cell_context.range_spec}")
            return False
        
        operation = {
            "type": "create_pivot_table",
            "range": cell_context.range_spec,
            "config": cell_context.pivot_config
        }
        
        self.pending_operations.append(operation)
        logger.info(f"Queued pivot table creation at {cell_context.range_spec}")
        
        return self._maybe_execute_batch()
    
    def _push_chart(self, cell_context: CellContext) -> bool:
        """Create chart from data"""
        # For now, just push the data values
        return self._push_values_only(cell_context)
    
    def _maybe_execute_batch(self) -> bool:
        """Execute batch if we've reached the batch size"""
        if len(self.pending_operations) >= self.batch_size:
            return self.execute_pending_batch()
        return True
    
    def execute_pending_batch(self) -> bool:
        """Execute all pending operations as a batch"""
        if not self.pending_operations:
            return True
        
        try:
            # Group operations by type for efficient batching
            value_operations = []
            note_operations = []
            pivot_operations = []
            
            for op in self.pending_operations:
                if op["type"] == "update_values":
                    value_operations.append(op)
                elif op["type"] == "add_note":
                    note_operations.append(op)
                elif op["type"] == "create_pivot_table":
                    pivot_operations.append(op)
            
            # Execute each type of operation
            success = True
            
            if value_operations:
                success &= self._execute_value_batch(value_operations)
            
            if note_operations:
                success &= self._execute_note_batch(note_operations)
            
            if pivot_operations:
                success &= self._execute_pivot_batch(pivot_operations)
            
            # Clear pending operations
            self.operation_count += len(self.pending_operations)
            self.pending_operations.clear()
            
            logger.info(f"Executed batch of operations. Total operations: {self.operation_count}")
            return success
            
        except Exception as e:
            logger.error(f"Error executing batch: {e}")
            return False
    
    def _execute_value_batch(self, operations: List[Dict[str, Any]]) -> bool:
        """Execute a batch of value update operations"""
        # In a real implementation, this would use Google Sheets batchUpdate API
        logger.info(f"Executing {len(operations)} value operations")
        
        for op in operations:
            range_spec = op["range"]
            values = op["values"]
            input_option = op["value_input_option"]
            
            # Simulate API call
            logger.debug(f"Updating {range_spec} with values: {values} (mode: {input_option})")
        
        return True
    
    def _execute_note_batch(self, operations: List[Dict[str, Any]]) -> bool:
        """Execute a batch of note addition operations"""
        logger.info(f"Executing {len(operations)} note operations")
        
        for op in operations:
            range_spec = op["range"]
            note = op["note"]
            
            # Simulate API call
            logger.debug(f"Adding note to {range_spec}: {note}")
        
        return True
    
    def _execute_pivot_batch(self, operations: List[Dict[str, Any]]) -> bool:
        """Execute a batch of pivot table creation operations"""
        logger.info(f"Executing {len(operations)} pivot operations")
        
        for op in operations:
            range_spec = op["range"]
            config = op["config"]
            
            # Simulate API call
            logger.debug(f"Creating pivot table at {range_spec} with config: {config}")
        
        return True
    
    def push_values(self, range_spec: str, values: List[Any]) -> bool:
        """Push raw values to a specific range"""
        context = CellContext(
            range_spec=range_spec,
            cell_type=CellType.SOURCE_DATA,
            values=values
        )
        return self.push_with_context(context)
    
    def push_formula(self, range_spec: str, formula: str) -> bool:
        """Push a formula to a specific range"""
        context = CellContext(
            range_spec=range_spec,
            cell_type=CellType.CALCULATION,
            formula=formula
        )
        return self.push_with_context(context)
    
    def push_complex_formula(self, range_spec: str, formula: str, computed_value: Any, note: Optional[str] = None) -> bool:
        """Push a complex formula with validation"""
        context = CellContext(
            range_spec=range_spec,
            cell_type=CellType.COMPLEX_CALCULATION,
            formula=formula,
            computed_value=computed_value,
            validation_note=note
        )
        return self.push_with_context(context)
    
    def add_note(self, range_spec: str, note: str) -> bool:
        """Add a note to a specific cell"""
        operation = {
            "type": "add_note",
            "range": range_spec,
            "note": note
        }
        
        self.pending_operations.append(operation)
        return self._maybe_execute_batch()
    
    def create_pivot_table(self, range_spec: str, pivot_config: Dict[str, Any]) -> bool:
        """Create a native Google Sheets pivot table"""
        context = CellContext(
            range_spec=range_spec,
            cell_type=CellType.PIVOT_TABLE,
            pivot_config=pivot_config
        )
        return self.push_with_context(context)
    
    def flush(self) -> bool:
        """Flush any pending operations"""
        return self.execute_pending_batch()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about operations performed"""
        return {
            "total_operations": self.operation_count,
            "pending_operations": len(self.pending_operations),
            "batch_size": self.batch_size,
            "notes_enabled": self.enable_notes
        }