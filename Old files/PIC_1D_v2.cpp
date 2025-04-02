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
    int neighbors[8];

    // Grid dimensions (excluding guard cells)
    int nx, ny;
};

// RankInfo: tracks tiles in each rank
struct RankInfo {
    int rank;
    int numTiles;
    vector<Tile> tiles;
};

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Each rank has 4 tiles
    RankInfo myRankInfo;
    myRankInfo.rank = rank;
    myRankInfo.numTiles = 4;
    myRankInfo.tiles.resize(myRankInfo.numTiles);

    int guard = 2;
    for (int t = 0; t < myRankInfo.numTiles; t++) {
        myRankInfo.tiles[t].nx = 10;
        myRankInfo.tiles[t].ny = 10;

        int totalX = myRankInfo.tiles[t].nx + 2 * guard;
        int totalY = myRankInfo.tiles[t].ny + 2 * guard;
        myRankInfo.tiles[t].grid.resize(totalX * totalY);

        // Initialize grid Ex based on rank + tile index
        for (int i = 0; i < totalX * totalY; i++) {
            myRankInfo.tiles[t].grid[i].Ex = rank + 0.1 * t;
        }
    }

    // Identify neighbors
    int leftNeighbor  = (rank == 0) ? size - 1 : rank - 1;
    int rightNeighbor = (rank == size - 1) ? 0 : rank + 1;

    // MPI Requests
    vector<MPI_Request> requests(2 * myRankInfo.numTiles);
    vector<vector<Grid> > recvBuffers(myRankInfo.numTiles);  // Store received data
    int reqCount = 0;

    // ---------- STAGE 1: POST ALL RECEIVES ----------
    for (int t = 0; t < myRankInfo.numTiles; t++) {
        int senderTile = (t == 0) ? 3 : (t - 1);
        int senderRank = (t == 0) ? leftNeighbor : rank;

        int totalX = myRankInfo.tiles[t].nx + 2 * guard;
        int totalY = myRankInfo.tiles[t].ny + 2 * guard;
        int columnSize = totalY * guard;

        recvBuffers[t].resize(columnSize);

        // Post receive
        MPI_Irecv(recvBuffers[t].data(), columnSize * sizeof(Grid), MPI_BYTE,
                  senderRank, senderTile, MPI_COMM_WORLD, &requests[reqCount]);
        reqCount++;
    }

    MPI_Barrier(MPI_COMM_WORLD);  // Ensure all receives are posted

    // ---------- STAGE 2: POST ALL SENDS ----------
    for (int t = 0; t < myRankInfo.numTiles; t++) {
        int receiverTile = (t == 3) ? 0 : t + 1;
        int receiverRank = (t == 3) ? rightNeighbor : rank;
        int senderTile = t; // My tile index is the tag

        int totalX = myRankInfo.tiles[t].nx + 2 * guard;
        int totalY = myRankInfo.tiles[t].ny + 2 * guard;
        int columnSize = totalY * guard;  // Increase buffer size for 2 columns

        vector<Grid> sendBuffer(columnSize);

        // Pack both leftmost guard columns
        for (int i = 0; i < totalY; i++) {
            for (int g = 0; g < guard; g++) {
                sendBuffer[i * guard + g] = myRankInfo.tiles[t].grid[i * totalX + g];
            }
        }

        // Post send
        MPI_Isend(sendBuffer.data(), columnSize * sizeof(Grid), MPI_BYTE,
                  receiverRank, senderTile, MPI_COMM_WORLD, &requests[reqCount]);
        reqCount++;
    }

    // Wait for all sends + receives
    MPI_Waitall(reqCount, requests.data(), MPI_STATUSES_IGNORE);

    // ---------- COPY RECEIVED BUFFERS INTO RIGHT GUARD COLUMN ----------
    for (int t = 0; t < myRankInfo.numTiles; t++) {
        int totalX = myRankInfo.tiles[t].nx + 2 * guard;
        int totalY = myRankInfo.tiles[t].ny + 2 * guard;

        // Copy received data into rightmost guard columns
        for (int i = 0; i < totalY; i++) {
            for (int g = 0; g < guard; g++) {
                int rightGuardColumn = totalX - guard + g;
                myRankInfo.tiles[t].grid[i * totalX + rightGuardColumn] = recvBuffers[t][i * guard + g];
            }
        }
    }

    // ---------- ORDERED PRINT (Rank 0 → Rank size-1) ----------
    MPI_Barrier(MPI_COMM_WORLD);
    for (int p = 0; p < size; p++) {
        if (rank == p) {
            cout << "\n===== OUTPUT FOR RANK " << rank << " =====\n";

            // Print receives
            for (int t = 0; t < myRankInfo.numTiles; t++) {
                int senderRank = (t == 0) ? leftNeighbor : rank;
                int senderTile = (t == 0) ? 3 : (t - 1);
                cout << "Tile " << t << " received from Rank " << senderRank
                     << ", Tile " << senderTile << endl;

                // Print first 3 Ex values from the right guard columns
                int totalX = myRankInfo.tiles[t].nx + 2 * guard;
                cout << "  Right guard column [Ex first 3 rows]: ";
                for (int row = 0; row < 3; row++) {
                    for (int g = 0; g < guard; g++) {
                        cout << myRankInfo.tiles[t].grid[row * totalX + (totalX - guard + g)].Ex << " ";
                    }
                    cout << " | ";
                }
                cout << "...\n";
            }

            // Print sends
            cout << "\nSends from Rank " << rank << ":\n";
            for (int t = 0; t < myRankInfo.numTiles; t++) {
                int receiverRank = (t == 3) ? rightNeighbor : rank;
                int receiverTile = (t == 3) ? 0 : t + 1;
                cout << "Tile " << t << " sent to Rank " << receiverRank
                     << ", Tile " << receiverTile << endl;
            }

            cout << flush;
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
}