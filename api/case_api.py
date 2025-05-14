"""
API endpoints for case management.
"""
import pandas as pd
from fastapi import APIRouter, HTTPException, Body
from typing import List, Dict, Any, Optional
from pydantic import BaseModel

from .case_manager import (
    get_cases, get_case, create_case, update_case,
    delete_case, create_anjishi_case, create_case_from_analysis
)
from .json_utils import sanitize_float_for_json

# Create router
router = APIRouter()

# Define models


class CaseMetadata(BaseModel):
    id: str
    title: str
    stockCode: str
    stockName: str
    createdAt: str
    updatedAt: str
    tags: List[str]


class CaseListResponse(BaseModel):
    cases: List[CaseMetadata]
    lastUpdated: str


class CaseCreateRequest(BaseModel):
    title: str
    stockCode: str
    stockName: str
    tags: Optional[List[str]] = None
    description: Optional[str] = None


class CaseUpdateRequest(BaseModel):
    title: Optional[str] = None
    tags: Optional[List[str]] = None
    description: Optional[str] = None


# Define routes
@router.get("/cases", response_model=CaseListResponse)
async def list_cases():
    """
    Get all cases.
    """
    cases = get_cases()
    # Sanitize the cases data to handle NaN values
    sanitized_cases = sanitize_float_for_json(cases)
    return {
        "cases": sanitized_cases,
        "lastUpdated": cases[0]["updatedAt"] if cases else ""
    }


@router.get("/cases/{case_id}")
async def get_case_by_id(case_id: str):
    """
    Get a specific case by ID.
    """
    case_data = get_case(case_id)
    if not case_data:
        raise HTTPException(status_code=404, detail="Case not found")

    # Sanitize the case data to handle NaN values
    sanitized_data = sanitize_float_for_json(case_data)
    return sanitized_data


@router.post("/cases")
async def create_new_case(case_data: Dict[str, Any] = Body(...)):
    """
    Create a new case.
    """
    try:
        result = create_case(case_data)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to create case: {str(e)}")


@router.put("/cases/{case_id}")
async def update_existing_case(case_id: str, case_data: Dict[str, Any] = Body(...)):
    """
    Update an existing case.
    """
    result = update_case(case_id, case_data)
    if not result:
        raise HTTPException(status_code=404, detail="Case not found")
    return result


@router.delete("/cases/{case_id}")
async def delete_existing_case(case_id: str):
    """
    Delete a case.
    """
    success = delete_case(case_id)
    if not success:
        raise HTTPException(status_code=404, detail="Case not found")
    return {"success": True}


@router.post("/cases/create-anjishi")
async def create_anjishi_case_endpoint():
    """
    Create a case for Anjishi (安记食品) based on our analysis.
    """
    try:
        result = create_anjishi_case()
        return result
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to create Anjishi case: {str(e)}")


class ExportCaseRequest(BaseModel):
    """
    Request model for exporting a stock to case library.
    """
    stockData: Dict[str, Any]
    analysisResult: Dict[str, Any]
    klineData: List[Dict[str, Any]]


@router.post("/cases/export")
async def export_to_case(request: ExportCaseRequest = Body(...)):
    """
    Export a stock analysis result to the case library.
    """
    try:
        # Convert klineData to DataFrame
        kline_df = pd.DataFrame(
            request.klineData) if request.klineData else pd.DataFrame()

        # Generate a title if not provided
        if "title" not in request.stockData:
            request.stockData["title"] = f"{request.stockData.get('name', '')}({request.stockData.get('code', '')})平台期分析"

        # Create case from analysis
        result = create_case_from_analysis(
            stock_data=request.stockData,
            analysis_result=request.analysisResult,
            kline_data=kline_df
        )

        return {"success": True, "case_id": result.get("id"), "message": "案例创建成功"}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to export case: {str(e)}")
