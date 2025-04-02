#include "Field_update.h"

/**
 * Assumptions on field staggering:
 *   - Ex is stored at grid index (i, j) but is physically at ((i+1/2)*dx, j*dy)
 *   - Ey is stored at grid index (i, j) but is physically at (i*dx, (j+1/2)*dy)
 *   - Ez is stored at grid index (i, j) and is physical at (i*dx, j*dy)
 *
 *   - Bx is stored at grid index (i, j) but is physical at (i*dx, (j+1/2)*dy)
 *   - By is stored at grid index (i, j) but is physical at ((i+1/2)*dx, j*dy)
 *   - Bz is stored at grid index (i, j) but is physical at ((i+1/2)*dx, (j+1/2)*dy)
 */

// Update B from time n-1/2 to n+1/2 using E^n
// The finite differences use forward differences so that the indices (j+1) and (i+1) remain valid
void updateBhalf(
    const vector<GridE>& Efull,    // E at time n
    const vector<GridB>& BhalfOld, // B at time n-1/2
    vector<GridB>& BhalfNew,       // B at time n+1/2 (output)
    int interior_nx, int interior_ny,
    int guard, double dt, double dx, double dy)
{
    int totalX = interior_nx + 2 * guard;
    int totalY = interior_ny + 2 * guard;

    // Loop over the interior cells for which the stencil is valid
    // Here, we update indices j = guard ... totalY - guard - 1 and i = guard ... totalX - guard - 1
    // This ensures that the accesses at (j+1) and (i+1) are within the grid
    for (int j = guard; j < totalY - guard - 1; j++) {
        for (int i = guard; i < totalX - guard - 1; i++) {
            int idx = j * totalX + i;
            // Update Bx at (i, j+1/2):
            BhalfNew[idx].Bx = BhalfOld[idx].Bx - (dt / dy) * (Efull[(j + 1) * totalX + i].Ez - Efull[idx].Ez);
            // Update By at (i+1/2, j):
            BhalfNew[idx].By = BhalfOld[idx].By + (dt / dx) * (Efull[j * totalX + (i + 1)].Ez - Efull[idx].Ez);
            // Update Bz at (i+1/2, j+1/2):
            BhalfNew[idx].Bz = BhalfOld[idx].Bz
                - (dt / dx) * (Efull[j * totalX + (i + 1)].Ey - Efull[idx].Ey)
                + (dt / dy) * (Efull[(j + 1) * totalX + i].Ex - Efull[idx].Ex);
        }
    }
}

// Update E from time n to n+1 using B^{n+1/2}
// The update uses backward differences so that the indices (j-1) and (i-1) are valid
void updateEfull(
    vector<GridE>& Efull,          // E at time n (will be updated to n+1)
    const vector<GridB>& Bhalf,    // B at time n+1/2
    int interior_nx, int interior_ny,
    int guard, double dt, double dx, double dy)
{
    int totalX = interior_nx + 2 * guard;
    int totalY = interior_ny + 2 * guard;

    // Loop over the interior cells for which the backward stencil is valid
    // We start at j = guard + 1 and i = guard + 1 so that (j-1) and (i-1) are within the interior
    for (int j = guard + 1; j < totalY - guard; j++) {
        for (int i = guard + 1; i < totalX - guard; i++) {
            int idx = j * totalX + i;
            // Update Ex at (i+1/2, j):
            Efull[idx].Ex += (dt / dy) * (Bhalf[idx].Bz - Bhalf[(j - 1) * totalX + i].Bz);
            // Update Ey at (i, j+1/2):
            Efull[idx].Ey -= (dt / dx) * (Bhalf[idx].Bz - Bhalf[j * totalX + (i - 1)].Bz);
            // Update Ez at (i, j):
            Efull[idx].Ez += (dt / dx) * (Bhalf[j * totalX + i].By - Bhalf[j * totalX + (i - 1)].By)
                           - (dt / dy) * (Bhalf[j * totalX + i].Bx - Bhalf[(j - 1) * totalX + i].Bx);
        }
    }
}
