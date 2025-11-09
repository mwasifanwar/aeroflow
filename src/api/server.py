from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import numpy as np

app = FastAPI(title="AeroFlow API", version="1.0.0")

class AirfoilRequest(BaseModel):
    parameters: List[float]
    airfoil_type: str = "parametric"
    points: int = 200

class SimulationRequest(BaseModel):
    airfoil_parameters: List[float]
    alpha: float = 5.0
    reynolds: float = 1e6
    mach: float = 0.0

class OptimizationRequest(BaseModel):
    initial_parameters: List[float]
    bounds: List[List[float]]
    alpha: float = 5.0
    method: str = "genetic"
    max_iterations: int = 100

class SurrogateTrainingRequest(BaseModel):
    design_parameters: List[List[float]]
    angles: List[float]
    coefficients: List[List[float]]

@app.post("/generate_airfoil")
async def generate_airfoil(request: AirfoilRequest):
    try:
        from src.geometry.airfoil_generator import AirfoilGenerator
        
        generator = AirfoilGenerator()
        
        if request.airfoil_type == "naca":
            digits = "".join(str(int(p * 10)) for p in request.parameters[:3])
            airfoil = generator.naca_4_digit(digits, request.points)
        else:
            airfoil = generator.parametric_airfoil(request.parameters, request.points)
        
        return {
            "status": "success",
            "airfoil_points": airfoil.tolist(),
            "parameters": request.parameters
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/run_simulation")
async def run_simulation(request: SimulationRequest):
    try:
        from src.geometry.airfoil_generator import AirfoilGenerator
        from src.geometry.mesh_generator import MeshGenerator
        from src.cfd.solver import CFDSolver
        from src.cfd.boundary_conditions import BoundaryConditions
        from src.cfd.post_processor import PostProcessor
        
        generator = AirfoilGenerator()
        airfoil = generator.parametric_airfoil(request.airfoil_parameters)
        
        mesh_gen = MeshGenerator()
        mesh = mesh_gen.generate_structured_mesh(airfoil)
        
        solver = CFDSolver()
        flow_variables = solver.solve_potential_flow(mesh, airfoil, request.alpha, request.mach)
        
        bc = BoundaryConditions()
        flow_variables = bc.apply_farfield_conditions(mesh, flow_variables, request.alpha)
        flow_variables = bc.apply_airfoil_conditions(mesh, airfoil, flow_variables)
        
        post_processor = PostProcessor()
        coefficients = post_processor.calculate_aerodynamic_coefficients(flow_variables, airfoil, mesh, request.alpha)
        
        return {
            "status": "success",
            "aerodynamic_coefficients": coefficients,
            "mesh_points": mesh.tolist(),
            "flow_variables": {k: v.tolist() for k, v in flow_variables.items()}
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/optimize_design")
async def optimize_design(request: OptimizationRequest):
    try:
        from src.neural_networks.surrogate_models import AerodynamicSurrogate
        from src.optimization.genetic_algorithm import GeneticOptimizer
        
        surrogate = AerodynamicSurrogate()
        
        dummy_designs = [request.initial_parameters] * 10
        dummy_angles = [request.alpha] * 10
        dummy_coeffs = [[0.5, 0.01, 0.0]] * 10
        
        surrogate.train(dummy_designs, dummy_angles, dummy_coeffs, epochs=100)
        
        optimizer = GeneticOptimizer()
        best_params, best_fitness = optimizer.optimize(surrogate, request.bounds, request.alpha)
        
        return {
            "status": "success",
            "optimized_parameters": best_params.tolist(),
            "best_fitness": best_fitness,
            "method": request.method
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/train_surrogate")
async def train_surrogate(request: SurrogateTrainingRequest):
    try:
        from src.neural_networks.surrogate_models import AerodynamicSurrogate
        
        surrogate = AerodynamicSurrogate()
        surrogate.train(request.design_parameters, request.angles, request.coefficients)
        
        return {
            "status": "success",
            "message": "Surrogate model trained successfully",
            "training_samples": len(request.design_parameters)
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "AeroFlow"}

if __name__ == "__main__":
    import uvicorn
    from src.utils.config import Config
    config = Config()
    uvicorn.run(app, host=config.get('api.host'), port=config.get('api.port'))