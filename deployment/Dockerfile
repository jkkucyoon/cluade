FROM python:3.11-slim

WORKDIR /app/deployment

# Install dependencies
COPY deployment/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY deployment/app.py .

# Copy model files from each project (must be generated first)
COPY project1_supplier_risk/models/ /app/project1_supplier_risk/models/
COPY project2_component_obsolescence/models/ /app/project2_component_obsolescence/models/
COPY project3_demand_forecasting/models/ /app/project3_demand_forecasting/models/

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run with uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
