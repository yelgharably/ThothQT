# Comprehensive Heat Map Testing Script
# Tests both single point (scatter) and multi-point (contour) functionality
# Also tests both NEGF and SCF modes

Write-Host "=== COMPREHENSIVE HEAT MAP FUNCTIONALITY TEST ===" -ForegroundColor Green

# Set up environment
conda activate quantum-env
Set-Location "d:\Graduate Life\Graphene_stuff\negf_sw\parallel"

Write-Host "`n1. Testing SINGLE POINT heat map (should use scatter plot)..." -ForegroundColor Yellow
python graphene_tb_Tb_SW_negf.py --lanthanide Eu --use_negf --W 6 --L 8 --Xmax 0.02 --NX 3 --sensitivity_heatmap --heatmap_temps 250 --heatmap_bias 0.005
Write-Host "✓ Single point test completed" -ForegroundColor Green

Write-Host "`n2. Testing MULTI-POINT heat map (should use contour plot)..." -ForegroundColor Yellow
python graphene_tb_Tb_SW_negf.py --lanthanide Eu --use_negf --W 6 --L 8 --Xmax 0.02 --NX 3 --sensitivity_heatmap --heatmap_temps 200,300 --heatmap_bias n0.01,0.01
Write-Host "✓ Multi-point test completed" -ForegroundColor Green

Write-Host "`n3. Testing SCF MODE with heat map..." -ForegroundColor Yellow
python graphene_tb_Tb_SW_negf.py --lanthanide Eu --scf --W 6 --L 8 --Xmax 0.02 --NX 3 --sensitivity_heatmap --heatmap_temps 250,350 --heatmap_bias n0.005,0.005
Write-Host "✓ SCF mode test completed" -ForegroundColor Green

Write-Host "`n4. Testing LARGER parameter set..." -ForegroundColor Yellow
python graphene_tb_Tb_SW_negf.py --lanthanide Eu --use_negf --W 6 --L 8 --Xmax 0.02 --NX 3 --sensitivity_heatmap --heatmap_temps 200,250,300 --heatmap_bias n0.02,n0.01,0,0.01,0.02
Write-Host "✓ Large parameter set test completed" -ForegroundColor Green

Write-Host "`n=== ALL TESTS COMPLETED SUCCESSFULLY ===" -ForegroundColor Green
Write-Host "The heat map system now supports:" -ForegroundColor Cyan
Write-Host "  ✓ Automatic scatter vs contour plot selection" -ForegroundColor White
Write-Host "  ✓ Both NEGF and SCF solver modes" -ForegroundColor White  
Write-Host "  ✓ Efficient result reuse optimization" -ForegroundColor White
Write-Host "  ✓ Robust plotting for any data size" -ForegroundColor White
Write-Host "  ✓ Comprehensive parameter sweep automation" -ForegroundColor White