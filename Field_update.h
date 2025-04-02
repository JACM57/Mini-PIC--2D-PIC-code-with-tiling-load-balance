#ifndef FIELD_UPDATE_H
#define FIELD_UPDATE_H

#include <vector>
using std::vector;


// GridE: Holds the electric field components (Ex, Ey, Ez)
struct GridE {
    double Ex, Ey, Ez;
};


// GridB: Holds the magnetic field components (Bx, By, Bz)
struct GridB {
    double Bx, By, Bz;
};

/**
 * Update B from time n-1/2 to n+1/2 using E at time n
 *
 *   Bx^{n+1/2} = Bx^{n-1/2} - (dt/dy)*[Ez^n(i,j+1) - Ez^n(i,j)]
 *   By^{n+1/2} = By^{n-1/2} + (dt/dx)*[Ez^n(i+1,j) - Ez^n(i,j)]
 *   Bz^{n+1/2} = Bz^{n-1/2} - (dt/dx)*[Ey^n(i+1,j) - Ey^n(i,j)] + (dt/dy)*[Ex^n(i,j+1) - Ex^n(i,j)]
 *      
 * @param Efull         E at time n
 * @param BhalfOld      B at time n-1/2 (input)
 * @param BhalfNew      B at time n+1/2 (output)
 * @param interior_nx   Number of interior cells in x
 * @param interior_ny   Number of interior cells in y
 * @param guard         Number of guard cells
 * @param dt            Time step
 * @param dx            Spacing in x
 * @param dy            Spacing in y
 */
void updateBhalf( const vector<GridE>& Efull, const vector<GridB>& BhalfOld, vector<GridB>& BhalfNew, int interior_nx, int interior_ny, int guard, double dt, double dx, double dy );

/**
 * Update E from time n to n+1 using B at time n+1/2.
 *
 *   Ex^{n+1} = Ex^n + (dt/dy)*[ Bz^{n+1/2}(i,j) - Bz^{n+1/2}(i,j-1) ]
 *   Ey^{n+1} = Ey^n - (dt/dx)*[ Bz^{n+1/2}(i,j) - Bz^{n+1/2}(i-1,j) ]
 *   Ez^{n+1} = Ez^n + (dt/dx)*[ By^{n+1/2}(i,j) - By^{n+1/2}(i-1,j) ] - (dt/dy)*[ Bx^{n+1/2}(i,j) - Bx^{n+1/2}(i,j-1) ]
 *              
 * @param Efull         E at time n (updated to time n+1)
 * @param Bhalf         B at time n+1/2
 * @param interior_nx   Number of interior cells in x
 * @param interior_ny   Number of interior cells in y
 * @param guard         Number of guard cells
 * @param dt            Time step
 * @param dx            Spacing in x
 * @param dy            Spacing in y
 */
void updateEfull( vector<GridE>& Efull, const vector<GridB>& Bhalf, int interior_nx, int interior_ny, int guard, double dt, double dx, double dy );

#endif // FIELD_UPDATE_H
