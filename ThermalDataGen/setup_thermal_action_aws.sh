#!/bin/bash
# Setup script for thermal action detection on AWS EC2
# Run this after extracting files on AWS instance

set -e  # Exit on error

echo "================================================================================"
echo "THERMAL ACTION DETECTION - AWS SETUP"
echo "================================================================================"
echo ""

# Navigate to project directory
cd /home/ec2-user/jingchen

echo "1. Installing Python dependencies..."
pip install --upgrade pip
pip install h5py numpy torch requests matplotlib pillow

echo ""
echo "2. Creating output directories..."
mkdir -p thermal_action_dataset/{frames,annotations,statistics/visualizations}
mkdir -p logs

echo ""
echo "3. Testing TDengine connection..."
if curl -s http://35.90.244.93:6041/rest/sql/thermal_sensors_pilot \
  -u root:taosdata \
  -d "SHOW TABLES" | grep -q "code\":0"; then
  echo "✅ TDengine connection successful!"
else
  echo "⚠️  TDengine connection failed - check network access"
fi

echo ""
echo "4. Verifying file structure..."
if [ -d "scripts/thermal_action" ]; then
  echo "✅ scripts/thermal_action/ found"
else
  echo "❌ scripts/thermal_action/ NOT found"
fi

if [ -d "DataAnnotationQA/src/data_pipeline" ]; then
  echo "✅ DataAnnotationQA/src/data_pipeline/ found"
else
  echo "❌ DataAnnotationQA/src/data_pipeline/ NOT found"
fi

if [ -d "DataAnnotationQA/Data/Gen3_Annotated_Data_MVP/Annotations" ]; then
  ann_count=$(ls DataAnnotationQA/Data/Gen3_Annotated_Data_MVP/Annotations/*.json 2>/dev/null | wc -l)
  echo "✅ Annotation files found: $ann_count files"
else
  echo "❌ Annotation files NOT found"
fi

echo ""
echo "================================================================================"
echo "SETUP COMPLETE!"
echo "================================================================================"
echo ""
echo "Next steps:"
echo "  1. Generate dataset:"
echo "     python scripts/thermal_action/generate_thermal_action_dataset.py \\"
echo "       --annotation-files DataAnnotationQA/Data/Gen3_Annotated_Data_MVP/Annotations/*.json \\"
echo "       --output-dir thermal_action_dataset \\"
echo "       --val-split 0.2"
echo ""
echo "  2. Validate dataset:"
echo "     python scripts/thermal_action/validate_dataset.py"
echo ""
echo "  3. Integrate with AlphAction:"
echo "     See AWS_EXECUTION_GUIDE.md"
echo ""
echo "================================================================================"

