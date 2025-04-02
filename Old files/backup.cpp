#include <mpi.h>
#include <iostream>
#include <vector>
#include <cassert>

using namespace std;

// Particle definition: tag, charge, positions (x,y,z) and momenta (px,py,pz)
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
    // Create a vector of particles
    vector<Particle> particles;

    // Create a 1D vector of grid cells, ordered in row-major format
    vector<Grid> grid;

    // Neighbors are identified by integers corresponding to their rank
    // Order: top, bottom, left, right, top-left, top-right, bottom-left, bottom-right
    // (first use the direct neighbors for updates and corrections before considering the diagonals)
    int neighbors[8];

    // Grid dimensions (excluding guard cells)
    int nx, ny;
};

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Create a tile for each MPI process
    Tile myTile;
    myTile.nx = 10;  // interior grid width
    myTile.ny = 10;  // interior grid height

    // Define number of guard cells (extra grid cells along the boundaries used
    // for communication with neighboring tiles)
    int guard = 1;

    // Total grid dimensions including guard cells on both sides
    int totalX = myTile.nx + 2 * guard;
    int totalY = myTile.ny + 2 * guard;
    myTile.grid.resize(totalX * totalY);

    // --------------- Test ---------------

    // Initialize grid cells with specific rank values
    for (int i = 0; i < totalX * totalY; i++) {
        // rank i -->  Ex = i
        myTile.grid[i].Ex = rank;
    }

    for (int i = 0; i < 8; i++) {
        myTile.neighbors[i] = -1;
    }

    // Exchange the left guard column with left neighbor and receive from right neighbor
    // [0 -> 1 -> 2 -> 3 -> 4 -> ... -> size - 1]
    // The left neighbor of rank 0 is the last rank (size - 1)
    int leftNeighbor = (rank == 0) ? size - 1 : rank - 1;
    // The right neighbor of the last rank is rank 0
    int rightNeighbor = (rank == size - 1) ? 0 : rank + 1;
    int tag = 0;

    int columnSize = totalY;
    vector<Grid> sendBuffer(columnSize);
    vector<Grid> recvBuffer(columnSize);

    // Pack the leftmost column (guard cell) from the grid
    // The index for row i and column 0 is: index = i * totalX + 0.
    for (int i = 0; i < columnSize; i++) {
        sendBuffer[i] = myTile.grid[i * totalX + 0];
    }

    // Use non-blocking MPI calls to send and receive the guard cell data
    MPI_Request reqs[2];
    MPI_Isend(sendBuffer.data(), columnSize * sizeof(Grid), MPI_BYTE,
              leftNeighbor, tag, MPI_COMM_WORLD, &reqs[0]);
    MPI_Irecv(recvBuffer.data(), columnSize * sizeof(Grid), MPI_BYTE,
              rightNeighbor, tag, MPI_COMM_WORLD, &reqs[1]);

    // Wait for send and receive to complete
    MPI_Waitall(2, reqs, MPI_STATUSES_IGNORE);

    // Unpack the received data into the rightmost guard cell column
    // The rightmost column index is totalX - guard
    int rightGuardColumn = totalX - guard;
    for (int i = 0; i < columnSize; i++) {
        myTile.grid[i * totalX + rightGuardColumn] = recvBuffer[i];
    }

    // Print and check the received guard column
    MPI_Barrier(MPI_COMM_WORLD);
    for (int p = 0; p < size; p++) {
        if (rank == p) {
            cout << "Rank " << rank 
                 << " received guard column from right neighbor (rank " << rightNeighbor << "):" << endl;
            for (int i = 0; i < columnSize; i++) {
                cout << "Row " << i << " Ex = " 
                     << myTile.grid[i * totalX + rightGuardColumn].Ex << endl;
            }
            cout << flush;
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
}
