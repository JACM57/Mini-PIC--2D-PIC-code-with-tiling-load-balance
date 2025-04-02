  #include <iostream>
  #include <vector>
  #include <fstream>

  using namespace std;

  //Grid initialization 
  const int Nx = 100, Ny = 100; //number of grid points
  const double dx = 1.0, dy = 1.0; //grid spacing
  const double dt = 1.0e-8;
  const int steps = 50;

  //Constants
  const double c = 2.99792e8;
  const double eps = 1/(c*376.7);
  const double mu = 376.7/c;

  //Field Arrays 

  //Electric Field
  double Ez[Nx][Ny] = {0};     // E_z at (i,j)
  double Ex[Nx + 1][Ny] = {0}; // E_x at (i+1/2, j)
  double Ey[Nx][Ny + 1] = {0}; // E_x at (i, j+1/2)

  //Magnetic Field
  double Bz[Nx + 1][Ny + 1] = {0};  // B_z at (i+1/2,j+1/2)
  double Bx[Nx][Ny + 1] = {0};      // B_x at (i, j+1/2)
  double By[Nx + 1][Ny] = {0};      // B_y at (i+1/2, j)

  //Update B (first half)
  void update_B_half(double Bz[][Ny + 1], double Bx[][Ny + 1], double By[][Ny], const double Ez[][Ny], const double Ex[][Ny], const double Ey[][Ny + 1], double dt, double dx, double dy){
    for (int i = 0; i < Nx; i++){
        for (int j = 1; j < Ny; j++){
            Bx[i][j] = Bx[i][j] - (dt/(2*dy))*(Ez[i][j] - Ez[i][j-1]);
        }
    }
    for (int i = 1; i < Nx; i++){
        for (int j = 0; j < Ny; j++){
            By[i][j] = By[i][j] + (dt/(2*dx))*(Ez[i][j] - Ez[i-1][j]); 
        }
    }
    for (int i = 1; i < Nx; i++){
        for (int j = 1; j < Ny; j++){
            Bz[i][j] = By[i][j] + (dt/(2*dx))*(Ey[i][j] - Ey[i-1][j]) - (dt/(2*dy))*(Ex[i][j] - Ex[i][j-1]); 
        }
    }
  }

  //Update E (full time step)
  void update_E(double Ez[][Ny], double Ex[][Ny], double Ey[][Ny + 1], const double Bz[][Ny + 1], const double Bx[][Ny + 1], const double By[][Ny], double dt, double dx, double dy, double eps, double mu){
    for (int i = 1; i < Nx; i++){
        for (int j = 0; j < Ny; j++){
            Ex[i][j] = Ex[i][j] + (dt/(eps*dy))*((Bz[i][j] - Bz[i][j-1])/mu);
        }
    }
    for (int i = 0; i < Nx; i++){
        for (int j = 1; j < Ny; j++){
            Ey[i][j] = Ey[i][j] - (dt/(eps*dx))*((Bz[i][j] - Bz[i-1][j])/mu); 
        }
    }
    for (int i = 0; i < Nx; i++){
        for (int j = 0; j < Ny; j++){
            Ez[i][j] = Ez[i][j] + (dt/(eps*dx))*((By[i][j] - By[i-1][j])/mu) - (dt/(eps*dy))*((Bx[i][j] - Bx[i][j-1])/mu);
        }

    }
  }

  //Update B (second half) *in case boundary conditions are needed*
  void update_B_final(double Bz[][Ny + 1], double Bx[][Ny + 1], double By[][Ny], const double Ez[][Ny], const double Ex[][Ny], const double Ey[][Ny + 1], double dt, double dx, double dy){
    for (int i = 0; i < Nx; i++){
        for (int j = 1; j < Ny; j++){
            Bx[i][j] = Bx[i][j] - (dt/(2*dy))*(Ez[i][j] - Ez[i][j-1]);
        }
    }
    for (int i = 1; i < Nx; i++){
        for (int j = 0; j < Ny; j++){
            By[i][j] = By[i][j] + (dt/(2*dx))*(Ez[i][j] - Ez[i-1][j]); 
        }
    }
    for (int i = 1; i < Nx; i++){
        for (int j = 1; j < Ny; j++){
            Bz[i][j] = By[i][j] + (dt/(2*dx))*(Ey[i][j] - Ey[i-1][j]) - (dt/(2*dy))*(Ex[i][j] - Ex[i][j-1]); 
        }
    }
  }  
  
  void save_field(const double field[Nx][Ny], string filename) {
    ofstream file(filename);
    if (file.is_open()) {
        for (int j = 0; j < Ny; j++) {
            file << field[Nx / 2][j] << "\n";  // Save center row
        }
        file.close();
    } else {
        cout << "Error: Unable to open file " << filename << endl;
    }
}

  //Time evolution
  void time_evolution(){
    double t = 0;

    // Save initial field
    save_field(Ez, "field_t0.txt");
    cout << "Saved: field_t0.txt" << endl;

    for (int step = 0; step < steps; step++){
        update_B_half(Bz,Bx,By,Ez,Ex,Ey,dt,dx,dy);
        update_E(Ez,Ex,Ey,Bz,Bx,By,dt,dx,dy,eps,mu);
        update_B_final(Bz,Bx,By,Ez,Ex,Ey,dt,dx,dy);
        t += dt;

        cout << "Step " << step << " completed, Ez[][] = " << Ez[50][50] << endl;

        // Save at 1/4, 1/2, and final step
        if (step == 25) save_field(Ez, "field_t1.txt");
        if (step == 50) save_field(Ez, "field_t2.txt");
        if (step == 99) save_field(Ez, "field_t3.txt");
    }
  }

  void initialize_TE_mode() {
    double A = 1.0;  // Amplitude
    double lambda = Nx/4.0; // Wavelength
    double k = (2 * M_PI)/lambda; // Wavenumber

    for (int i = 0; i < Nx; i++) {
        for (int j = 0; j < Ny; j++) {
            double x = i * dx;
            double y = j * dy;
            Ez[i][j] = A * sin(k * x) * sin(k * y); // TE mode: Ez sine wave
            Bx[i][j] = 0.0;
            By[i][j] = 0.0;
        }
    }
}

void initialize_TM_mode() {
    double A = 1.0;  // Amplitude
    double lambda = Nx / 2.0; // Wavelength
    double k = 2 * M_PI / lambda; // Wavenumber

    for (int i = 0; i < Nx; i++) {
        for (int j = 0; j < Ny; j++) {
            double x = i * dx;
            double y = j * dy;
            Bz[i][j] = A * sin(k * x) * sin(k * y); // TM mode: Bz sine wave
            Ex[i][j] = 0.0;
            Ey[i][j] = 0.0;
        }
    }
}

int main() {
    initialize_TE_mode();

    time_evolution();

    return 0;
}

