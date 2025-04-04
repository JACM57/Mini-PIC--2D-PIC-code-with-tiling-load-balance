#include "Field_update.h"

/**
 * Ex -> Stored at code index (i, j) || Physically located at (x, y) = ((i + 1/2) * dx, j * dy)
 * Ey -> Stored at code index (i, j) || Physically located at (x, y) = (i * dx, (j + 1/2) * dy)
 * Ez -> Stored at code index (i, j) || Physically located at (x, y) = (i * dx, j * dy)
 * 
 * Bx -> Stored at code index (i, j) || Physically located at (x, y) = (i * dx, (j + 1/2) * dy)
 * By -> Stored at code index (i, j) || Physically located at (x, y) = ((i + 1/2) * dx, j * dy)
 * Bz -> Stored at code index (i, j) || Physically located at (x, y) = ((i + 1/2) * dx, (j + 1/2) * dy)
 */

// Update B from time n-1/2 to n+1/2 using E^n
void updateBhalf(
    const vector<GridE>& Efull,    // E at time n
    const vector<GridB>& BhalfOld, // B at time n-1/2
    vector<GridB>& BhalfNew,       // B at time n+1/2 (output)
    int interior_nx, int interior_ny,
    int guard, double dt, double dx, double dy)
{
    int totalX = interior_nx + 2 * guard;
    int totalY = interior_ny + 2 * guard;

    // Update every interior cell
    // Note: We assume that guard cell data (for j+1 or i+1 at the boundary) have been set via communication
    for (int j = guard; j < totalY - guard; j++) {
        for (int i = guard; i < totalX - guard; i++) {
            int idx = j * totalX + i;
            BhalfNew[idx].Bx = BhalfOld[idx].Bx - (dt / (2*dy)) * (Efull[(j+1) * totalX + i].Ez - Efull[idx].Ez);
            BhalfNew[idx].By = BhalfOld[idx].By + (dt / (2*dx)) * (Efull[j * totalX + (i+1)].Ez - Efull[idx].Ez);
            BhalfNew[idx].Bz = BhalfOld[idx].Bz
                - (dt / (2*dx)) * (Efull[j * totalX + (i+1)].Ey - Efull[idx].Ey)
                + (dt / (2*dy)) * (Efull[(j+1) * totalX + i].Ex - Efull[idx].Ex);
        }
    }
}


// Update E from time n to n+1 using B^{n+1/2}
void updateEfull(
    vector<GridE>& Efull,          // E at time n (will be updated to n+1)
    const vector<GridB>& Bhalf,    // B at time n+1/2
    int interior_nx, int interior_ny,
    int guard, double dt, double dx, double dy)
{
    int totalX = interior_nx + 2*guard;
    int totalY = interior_ny + 2*guard;

    // Loop over interior cells
    for(int j = guard; j < totalY - guard; j++) {
        for(int i = guard; i < totalX - guard; i++) {
            int idx = j*totalX + i;
            Efull[idx].Ex += (dt / dy) * (Bhalf[idx].Bz - Bhalf[(j-1)*totalX + i].Bz);
            Efull[idx].Ey -= (dt / dx) * (Bhalf[idx].Bz - Bhalf[j*totalX + (i-1)].Bz);
            Efull[idx].Ez += (dt / dx) * (Bhalf[j*totalX + i].By - Bhalf[j*totalX + (i-1)].By)
                - (dt / dy) * (Bhalf[j*totalX + i].Bx - Bhalf[(j-1)*totalX + i].Bx);
        }
    }
}
