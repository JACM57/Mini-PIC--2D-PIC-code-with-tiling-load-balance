#include <mpi.h>
#include <iostream>
#include <vector>
#include <cassert>

using namespace std;

// Particle definition (unused in this test)
struct Particle {
    int tag;
    double q;
    double x, y, z;
    double px, py, pz;
};

// Grid cell definition: Electric and Magnetic field components
struct Grid {
    double Ex, Ey, Ez;
    double Bx, By, Bz;
};

// Tile structure: contains particles, grid cells, and neighbor information
struct Tile {
    // Vector of particles (unused here)
    vector<Particle> particles;
    
    // 1D vector for grid cells (row-major order)
    vector<Grid> grid;
    
    // Neighbors: order: top, bottom, left, right, top-left, top-right, bottom-left, bottom-right
    int neighbors[8];
    
    // Interior grid dimensions (excluding guard cells)
    int nx, ny;
};

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    int worldRank, worldSize;
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

    // --- Create a 2D Cartesian Topology ---
    int dims[2] = {0, 0};
    MPI_Dims_create(worldSize, 2, dims);  // Let MPI choose dims that multiply to worldSize

    int periods[2] = {1, 1}; // periodic boundaries in both dimensions
    MPI_Comm cartComm;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 1, &cartComm);

    int cartRank;
    MPI_Comm_rank(cartComm, &cartRank);
    int coords[2];
    MPI_Cart_coords(cartComm, cartRank, 2, coords);

    // Determine neighbors using MPI_Cart_shift.
    int leftNeighbor, rightNeighbor, topNeighbor, bottomNeighbor;
    MPI_Cart_shift(cartComm, 1, 1, &leftNeighbor, &rightNeighbor); // horizontal (dimension 1)
    MPI_Cart_shift(cartComm, 0, 1, &topNeighbor, &bottomNeighbor); // vertical (dimension 0)

    // Print Cartesian info for debugging.
    MPI_Barrier(cartComm);
    if(cartRank == 0) {
        cout << "Cartesian grid dimensions: " << dims[0] << " x " << dims[1] << endl;
    }
    MPI_Barrier(cartComm);
    for (int p = 0; p < worldSize; p++) {
        if (cartRank == p) {
            cout << "CartRank " << cartRank << " at coords (" 
                 << coords[0] << ", " << coords[1] << ") "
                 << "neighbors: left=" << leftNeighbor 
                 << ", right=" << rightNeighbor 
                 << ", top=" << topNeighbor 
                 << ", bottom=" << bottomNeighbor << endl;
            cout << flush;
        }
        MPI_Barrier(cartComm);
    }

    // --- Setup the Tile and Grid ---
    Tile myTile;
    myTile.nx = 10;  // interior grid width
    myTile.ny = 10;  // interior grid height

    int guard = 1; // number of guard cells on each side
    int totalX = myTile.nx + 2 * guard; // total columns
    int totalY = myTile.ny + 2 * guard; // total rows
    myTile.grid.resize(totalX * totalY);

    // Initialize every grid cell's Ex to the Cartesian rank (for identification)
    for (int i = 0; i < totalX * totalY; i++) {
        myTile.grid[i].Ex = cartRank;
    }
    for (int i = 0; i < 8; i++) {
        myTile.neighbors[i] = -1; // placeholder; not used in this test
    }

    // --- Horizontal (Left/Right) Communication Test ---
    // Use row 0 for horizontal communication.
    int testRowHor = 0;
    vector<Grid> sendBufferHor(1);
    vector<Grid> recvBufferHor(1);
    // Pack the leftmost cell of row 0 (col 0)
    sendBufferHor[0] = myTile.grid[testRowHor * totalX + 0];
    int tagHor = 100;
    MPI_Request reqsHor[2];
    MPI_Isend(sendBufferHor.data(), 1 * sizeof(Grid), MPI_BYTE,
              leftNeighbor, tagHor, cartComm, &reqsHor[0]);
    MPI_Irecv(recvBufferHor.data(), 1 * sizeof(Grid), MPI_BYTE,
              rightNeighbor, tagHor, cartComm, &reqsHor[1]);
    MPI_Waitall(2, reqsHor, MPI_STATUSES_IGNORE);
    // Unpack into the right guard cell of row 0 (col index totalX - guard)
    int destColHor = totalX - guard;
    myTile.grid[testRowHor * totalX + destColHor] = recvBufferHor[0];

    // --- Vertical (Up/Down) Communication Test ---
    // Use row totalY - guard (i.e. the bottom guard row, which is row index totalY - 1)
    int testRowVert = totalY - guard;
    vector<Grid> sendBufferVert(1);
    vector<Grid> recvBufferVert(1);
    // Pack the leftmost cell of the bottom row (col 0)
    sendBufferVert[0] = myTile.grid[testRowVert * totalX + 0];
    int tagVert = 200;
    MPI_Request reqsVert[2];
    MPI_Isend(sendBufferVert.data(), 1 * sizeof(Grid), MPI_BYTE,
              topNeighbor, tagVert, cartComm, &reqsVert[0]);
    MPI_Irecv(recvBufferVert.data(), 1 * sizeof(Grid), MPI_BYTE,
              bottomNeighbor, tagVert, cartComm, &reqsVert[1]);
    MPI_Waitall(2, reqsVert, MPI_STATUSES_IGNORE);
    // Unpack into the right guard cell of the bottom row (col index totalX - guard)
    int destColVert = totalX - guard;
    myTile.grid[testRowVert * totalX + destColVert] = recvBufferVert[0];

    // --- Synchronized Debug Output ---
    // Print horizontal communication result (row 0)
    MPI_Barrier(cartComm);
    for (int p = 0; p < worldSize; p++) {
        if (cartRank == p) {
            cout << "\n[Horizontal Test] CartRank " << cartRank 
                 << " (coords: " << coords[0] << "," << coords[1] 
                 << ") - row " << testRowHor << ", right guard cell (col " << destColHor << ")"
                 << " now has Ex = " 
                 << myTile.grid[testRowHor * totalX + destColHor].Ex << endl;
            cout << flush;
        }
        MPI_Barrier(cartComm);
    }

    // Print vertical communication result (bottom row)
    MPI_Barrier(cartComm);
    for (int p = 0; p < worldSize; p++) {
        if (cartRank == p) {
            cout << "\n[Vertical Test] CartRank " << cartRank 
                 << " (coords: " << coords[0] << "," << coords[1] 
                 << ") - bottom row (row " << testRowVert << "), right guard cell (col " << destColVert << ")"
                 << " now has Ex = " 
                 << myTile.grid[testRowVert * totalX + destColVert].Ex << endl;
            cout << flush;
        }
        MPI_Barrier(cartComm);
    }

    MPI_Finalize();
    return 0;
}
