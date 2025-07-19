# backend/app/routers/workflow.py
from fastapi import APIRouter, HTTPException, Depends, Path, Query
from typing import List, Optional

from ..services.strategy_runner import strategy_runner_service, StrategyRunnerService
from ..models.workflow import (
    WorkflowConfig,
    WorkflowRunResponse,
    WorkflowStatusResponse,
    WorkflowResultResponse,
    DefaultConfigResponse,
    RunHistoryResponse,
    RunHistoryItem
)
import logging

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/workflow",
    tags=["Workflow"],
)

@router.post("/run", response_model=WorkflowRunResponse, status_code=202)
async def run_workflow(
    config: WorkflowConfig,
    runner: StrategyRunnerService = Depends(lambda: strategy_runner_service)
):
    """Starts a new strategy workflow run."""
    logger.info(f"Request to run workflow. Name: '{config.strategy_name}'")
    try:
        # Service now returns the WorkflowRunResponse directly
        response_data = runner.start_workflow(config)
        logger.info(f"Workflow task {response_data.task_id} queued for Run ID {response_data.run_id}.")
        return response_data # Return the response model {message, task_id, run_id}
    except Exception as e:
        logger.error(f"Failed to start workflow task: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to start workflow: {str(e)}")

@router.get("/status/{run_id}", response_model=WorkflowStatusResponse)
async def get_workflow_status(
    run_id: str = Path(..., description="The custom ID of the workflow run"), # Use run_id
    runner: StrategyRunnerService = Depends(lambda: strategy_runner_service)
):
    """Checks the status of a specific workflow run using its custom Run ID."""
    logger.debug(f"Request status for Run ID: {run_id}")
    status = runner.get_workflow_status(run_id) # Service uses run_id to check Celery task
    if not status:
        logger.warning(f"Status check for Run ID {run_id} returned None.")
        raise HTTPException(status_code=404, detail="Workflow run status could not be determined.")
    logger.debug(f"Returning status for Run ID {run_id}: {status.status}")
    return status

@router.get("/results/{run_id}", response_model=WorkflowResultResponse)
async def get_workflow_results(
    run_id: str = Path(..., description="The custom ID of the workflow run"), # Use run_id
    runner: StrategyRunnerService = Depends(lambda: strategy_runner_service)
):
    """Retrieves the results of a completed workflow run using its custom Run ID."""
    logger.debug(f"Request results for Run ID: {run_id}")
    results = runner.get_workflow_results(run_id) # Service uses run_id

    if results is None:
        status_check = runner.get_workflow_status(run_id)
        if status_check.status in ['PENDING', 'STARTED', 'PROGRESS', 'RETRY']:
             logger.info(f"Results requested for run {run_id}, but status is {status_check.status}.")
             raise HTTPException(status_code=202, detail=f"Workflow run '{run_id}' is not finished yet. Status: {status_check.status}")
        else:
             logger.warning(f"Results requested for run {run_id}, but status is {status_check.status} and no results found.")
             raise HTTPException(status_code=404, detail=f"Workflow run '{run_id}' not found or results unavailable (Status: {status_check.status}).")

    logger.info(f"Returning results for Run ID {run_id}. Status: {results.status}")
    return results

@router.get("/config/defaults", response_model=DefaultConfigResponse)
async def get_default_configuration(
    runner: StrategyRunnerService = Depends(lambda: strategy_runner_service)
):
    """Returns the default configuration parameters."""
    logger.debug("Request default configuration.")
    defaults = runner.get_default_config()
    return DefaultConfigResponse(defaults=defaults)

@router.get("/history", response_model=RunHistoryResponse)
async def get_run_history(
    limit: int = Query(50, ge=1, le=200, description="Maximum number of history items to return"),
    runner: StrategyRunnerService = Depends(lambda: strategy_runner_service)
):
    """ Gets a list of past workflow runs, sorted by most recent completion time. """
    logger.info(f"Request run history (limit: {limit}).")
    try:
        history_data = runner.get_history(limit=limit)
        logger.info(f"Retrieved {len(history_data)} history items.")
        return RunHistoryResponse(history=history_data)
    except Exception as e:
        logger.error(f"Error retrieving run history: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to retrieve run history.")