#include <cstdlib>
#include <cmath>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "global.h"
#include "multigrid.h"

float bias = 1.0f;
float beta = 0.0f;
struct Box box;

void myexit(const int flag) {
  std::cout.flush();
  std::cerr.flush();
  std::exit(flag);
}

void myexception(const std::exception &e) {
  std::cout << ": Exception: " << e.what() << std::endl;
  std::cout.flush();
  std::cerr.flush();
  std::exit(1);
}

static struct particle fill_particle_chi(const double ra, const double dec,
                                         const double chi,
                                         const double wt) {
  struct particle curp;
  if (!(chi > 0.0) || !std::isfinite(chi)) {
    std::cout << "chi=" << chi << " out of range." << std::endl;
    myexit(1);
  }
  if (dec < -90.0 || dec > 90.0) {
    std::cout << "DEC=" << dec << " out of range." << std::endl;
    myexit(1);
  }
  const double theta = (90.0 - dec) * M_PI / 180.0;
  const double phi = (ra)*M_PI / 180.0;
  curp.pos[0] = chi * std::sin(theta) * std::cos(phi);
  curp.pos[1] = chi * std::sin(theta) * std::sin(phi);
  curp.pos[2] = chi * std::cos(theta);
  curp.wt = wt;
  return (curp);
}

static std::vector<struct particle> read_data_chi(const char fname[]) {
  std::ifstream fs(fname);
  if (!fs) {
    std::cerr << "Unable to open " << fname << " for reading." << std::endl;
    myexit(1);
  }
  std::string buf;
  do {
    getline(fs, buf);
  } while (!fs.eof() && buf[0] == '#');

  std::vector<struct particle> P;
  try {
    P.reserve(10000000);
  } catch (std::exception &e) {
    myexception(e);
  }
  while (!fs.eof()) {
    double ra, dec, chi, wt;
    std::istringstream(buf) >> ra >> dec >> chi >> wt;
    struct particle curp = fill_particle_chi(ra, dec, chi, wt);
    try {
      P.push_back(curp);
    } catch (std::exception &e) {
      myexception(e);
    }
    getline(fs, buf);
  }
  fs.close();
  return (P);
}

int main(int argc, char **argv) {
  if (argc != 8 && argc != 9) {
    std::cout << "Usage: recon_mw <data-file> <random-file> <bias> <f-growth> "
                 "<R-filter> <OmegaM> <random_rsd:0|1> [input_mode:z|chi]"
              << std::endl;
    myexit(1);
  }

  const char *data_file = argv[1];
  const char *random_file = argv[2];
  bias = std::atof(argv[3]);
  const float f_growth = std::atof(argv[4]);
  beta = f_growth / (bias + 1e-30f);
  const float Rf = std::atof(argv[5]);
  const double omega_m = std::atof(argv[6]);
  const int random_rsd = std::atoi(argv[7]);
  if (!((random_rsd == 0) || (random_rsd == 1))) {
    std::cerr << "random_rsd must be 0 or 1" << std::endl;
    myexit(1);
  }

  std::string input_mode = "z";
  if (argc == 9) {
    input_mode = argv[8];
  }
  if (!(input_mode == "z" || input_mode == "chi")) {
    std::cerr << "input_mode must be z or chi" << std::endl;
    myexit(1);
  }

  LCDM lcdm(omega_m);

  std::vector<struct particle> D =
      (input_mode == "chi") ? read_data_chi(data_file) : read_data(data_file, lcdm);
  std::vector<struct particle> R =
      (input_mode == "chi") ? read_data_chi(random_file) : read_data(random_file, lcdm);
  std::vector<struct particle> R1 = R; // selection function
  std::vector<struct particle> R2 = R; // shifted field

  std::cout << "# Read " << D.size() << " objects from " << data_file << std::endl;
  std::cout << "# Read " << R.size() << " randoms from " << random_file
            << " (duplicated for selection+shift)" << std::endl;
  std::cout << "# Input mode: " << input_mode << std::endl;

  remap_pos(D, R1, R2);
  std::cout << "# Enclosing survey in a box of side " << box.L << " Mpc/h." << std::endl;
  std::cout << "# Grid/mesh size is " << box.L / Ng << " Mpc/h"
            << " and filter scale is " << Rf << " Mpc/h." << std::endl;

  std::vector<float> delta = make_grid(D, R1, Rf);
  std::vector<float> phi = MultiGrid::fmg(delta, Ng);

  // Displaced galaxies (with RSD term when f_growth>0).
  shift_obj(D, phi);

  // Shifted randoms: Padmanabhan 2012 default is "no RSD term" for randoms.
  if (random_rsd == 0) {
    const float beta_saved = beta;
    beta = 0.0f;
    shift_obj(R2, phi);
    beta = beta_saved;
  } else {
    shift_obj(R2, phi);
  }

  write_data(D, "data_rec.xyzw");
  write_data(R2, "rand_rec.xyzw");
  return 0;
}
