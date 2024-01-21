#include <igl/read_triangle_mesh.h>
#include <igl/write_triangle_mesh.h>
#include <igl/opengl/glfw/Viewer.h>
#include <igl/triangulated_grid.h>
#include <igl/boundary_facets.h>
#include <igl/random_quaternion.h>
#include <igl/copyleft/cgal/mesh_boolean.h>
#include <igl/copyleft/cgal/intersect_other.h>
#include <igl/centroid.h>
#include <igl/connected_components.h>
#include <igl/adjacency_matrix.h>
#include <igl/per_vertex_normals.h>
#include <igl/barycentric_interpolation.h>
#include <igl/blue_noise.h>
#include <igl/doublearea.h>
#include <igl/PI.h>
#include <igl/volume.h>
#include <cmath>
#include <random>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <filesystem>

double planar_function(
  const double a,
  const double b,
  const double c,
  const double v1,
  const double v2)
{
  return a * v1 + b * v2 + c;
}

double paraboloidal_function(
  const double a,
  const double b,
  const double c,
  const double c_v1,
  const double c_v2,
  const double v1,
  const double v2)
{
  return a * (v1 - c_v1) * (v1 - c_v1) + b * (v2 - c_v2) * (v2 - c_v2) + c;
}

double sinusoidal_function(
  const double a,
  const double b,
  const double c,
  const double w,
  const double h,
  const double v1,
  const double v2)
{
  return w * sin(a * v1 + b * v2 + c) + h;
}

double square_function(
  const double h,
  const double t,
  const double c_v,
  const double v)
{
  int r;
  if (v - c_v >= 0)
  {
    r = (v - c_v) / t;
    if (v - c_v - r * t <= t/2)
      return h;
    else
      return 0;
  }
  else
  {
    r = (c_v - v) / t;
    if (-v + c_v - r * t >= t/2)
      return h;
    else
      return 0;
  }
}

double rectangular_pulse_function(
  const double h,
  const double w1,
  const double w2,
  const double c_v1,
  const double c_v2,
  const double v1,
  const double v2)
{
  if ((v1 - c_v1 <= w1/2) && (v1 - c_v1 >= -w1/2) && (v2 - c_v2 <= w2/2) && (v2 - c_v2 >= -w2/2))
    return h;
  else
    return 0;
}

void generate_hyperplane(
  const int n,
  const std::string cut_type,
  Eigen::MatrixXd & V,
  Eigen::MatrixXi & F)
{
  igl::triangulated_grid(n,n,V,F);
  // center at origin
  V.array() -= 0.5;
  V.conservativeResize(V.rows(),3);
  if (cut_type == "planar")
  {
    double a, b, c;
    a = 5.0;
    b = 8.0;
    c = 0.0;
    std::cout << "Plane: z = " << a << "x + " << b << "y + " << c << "\n";
    for(int i = 0; i<V.rows(); i++)
      V(i,2) = planar_function(a,b,c,V(i,0),V(i,1)); // z = a x + b y + c
  }
  else if (cut_type == "sine")
  {
    double a, b, c, w, h;
    // a = 50.0;
    // b = 50.0; 
    // c = 0.0;
    // w = 0.01; 
    // h = 0.0;
    a = 40.0 + static_cast <double> (rand()) / (static_cast <double> (RAND_MAX/20.0));
    b = 40.0 + static_cast <double> (rand()) / (static_cast <double> (RAND_MAX/20.0));
    c = 0.0;
    w = 0.01 + static_cast <double> (rand()) / (static_cast <double> (RAND_MAX/0.01));
    h = 0.0;
    std::cout << "Sinusoid: z = " << w << " sin(" << a << "x + " << b << "y + " << c << ") + " << h << "\n";
    for(int i = 0; i<V.rows(); i++)
      V(i,2) = sinusoidal_function(a,b,c,w,h,V(i,0),V(i,1)); // z = w sin(a x + b y + c) + h
  }
  else if (cut_type == "parabolic")
  {
    double a, b, c, c_v0, c_v1;
    // a = 5.0;
    // b = 20.0;
    // c = 0.0;
    // c_v0 = 0.0; // center coordinate of v0
    // c_v1 = 0.0; // center coordinate of v1
    a = 5.0 + static_cast <double> (rand()) / (static_cast <double> (RAND_MAX/15.0));
    b = 5.0 + static_cast <double> (rand()) / (static_cast <double> (RAND_MAX/15.0));
    c = 0.0;
    c_v0 = 0.0; // center coordinate of v0
    c_v1 = 0.0; // center coordinate of v1
    std::cout << "Paraboloid: z = " << a << "(x - " << c_v0 << ")^2 + " << b << "(y - " << c_v1 << ")^2 + " << c << "\n";
    for(int i = 0; i<V.rows(); i++)
      V(i,2) = paraboloidal_function(a,b,c,c_v0,c_v1,V(i,0),V(i,1)); // z = a (x - c_x)^2 + b (y - c_y)^2 + c
  }
  else if (cut_type == "square")
  {
    double h, t, c;
    // h = 0.02; // height of the pulse
    // t = 0.1;  // pulse period 
    // c = 0.0;  // center coordinate 
    h = 0.01 + static_cast <double> (rand()) / (static_cast <double> (RAND_MAX/0.09)); 
    t = 0.1 + static_cast <double> (rand()) / (static_cast <double> (RAND_MAX/0.1)); 
    c = 0.0;  // center coordinate 
    std::cout << "Square: height = " << h << ", period = " << t << ", center = " << c << "\n";
    for(int i = 0; i<V.rows(); i++)
      V(i,2) = square_function(h,t,c,V(i,0));
  }
  else if (cut_type == "pulse")
  {
    double h, w0, w1, c_v0, c_v1;
    // h = 0.05;   // 0.01 ~ 0.1
    // w0 = 0.03;  // 0.05 ~ 0.4
    // w1 = 0.03;  // 0.05 ~ 0.4
    // c_v0 = 0.0; 
    // c_v1 = 0.0;
    h = 0.01 + static_cast <double> (rand()) / (static_cast <double> (RAND_MAX/0.09));
    w0 = 0.05 + static_cast <double> (rand()) / (static_cast <double> (RAND_MAX/0.05));
    w1 = 0.05 + static_cast <double> (rand()) / (static_cast <double> (RAND_MAX/0.05));
    c_v0 = 0.0; 
    c_v1 = 0.0;
    std::cout << "Rectangular pulse: height = " << h << ", period 0 = " << w0 << ", period 1 = " << w1;
    std::cout << ", center 0 = " << c_v0 << ", center 1 = " << c_v1 << "\n";
    for(int i = 0; i<V.rows(); i++)
      V(i,2) = rectangular_pulse_function(h,w0,w1,c_v0,c_v1,V(i,0),V(i,1));
  }
  //////////////////////////////////////////////////////////////////////////
  // grab the boundary before duplicating facets
  Eigen::MatrixXi O;
  igl::boundary_facets(F,O);
  // Glue grid to copy of itself to make solid
  F.conservativeResize(F.rows()*2,F.cols());
  F.bottomRows(F.rows()/2) = ((F.topRows(F.rows()/2)).array() + V.rows()).rowwise().reverse();
  F = F.rowwise().reverse().eval();
  V.conservativeResize(V.rows()*2,V.cols());
  V.bottomRows(V.rows()/2) = V.topRows(V.rows()/2).rowwise() + Eigen::RowVector3d(0,0,1);
  // number of non boundary facets
  const int m = F.rows();
  // add two more for each boundary
  F.conservativeResize(F.rows()+2*O.rows(),F.cols());
  for(int i = 0;i<O.rows();i++)
  {
    F.row(m+i*2+0)<< O(i,0),O(i,1),V.rows()/2+O(i,1);
    F.row(m+i*2+1)<< V.rows()/2+O(i,1),V.rows()/2+O(i,0),O(i,0);
  }
}

void recenter_mesh(
  Eigen::MatrixXd & V,
  Eigen::MatrixXi & F)
{
  Eigen::RowVector3d C; // centroid of V
  igl::centroid(V,F,C);
  V.rowwise() -= C;
}

void compute_number_of_components(
  const Eigen::MatrixXi & F,
  int & num_of_components)
{
  Eigen::SparseMatrix<int> adjacency_matrix;
  igl::adjacency_matrix(F, adjacency_matrix);
  Eigen::MatrixXi C, K;
  igl::connected_components(adjacency_matrix, C, K);
  num_of_components = K.rows();
}

void get_mesh_volume(
  const Eigen::MatrixXd & V,
  const Eigen::MatrixXi & F,
  double & vol)
{
  Eigen::MatrixXd V2(V.rows() + 1, V.cols());
  V2.topRows(V.rows()) = V;
  V2.bottomRows(1).setZero();
  Eigen::MatrixXi T(F.rows(), 4);
  T.leftCols(3) = F;
  T.rightCols(1).setConstant(V.rows());
  Eigen::VectorXd tet_vol;
  igl::volume(V2, T, tet_vol);
  vol = std::abs(tet_vol.sum());
}

void sample_pc_with_blue_noise(
  const int & num_points,
  const Eigen::MatrixXd & mesh_v,
  const Eigen::MatrixXi & mesh_f,
  Eigen::MatrixXd & pc,
  Eigen::MatrixXd & normals)
{
  Eigen::VectorXd A;
  igl::doublearea(mesh_v, mesh_f, A);
  const double radius = sqrt(((A.sum()*0.5/(num_points*0.6162910373))/igl::PI));
  std::cout << "Blue noise radius: " << radius << "\n";
  Eigen::MatrixXd B;
  Eigen::VectorXi I;
  igl::blue_noise(mesh_v, mesh_f, radius, B, I, pc);
  Eigen::MatrixXd vertex_normals;
  igl::per_vertex_normals(mesh_v, mesh_f, vertex_normals);
  igl::barycentric_interpolation(vertex_normals, mesh_f, B, I, normals);
  normals.rowwise().normalize();
}

const static Eigen::IOFormat CSVFormat(Eigen::StreamPrecision, Eigen::DontAlignCols, ",", "\n");

void write_matrix_to_csv(
  const std::string & filename,
  Eigen::MatrixXd & T)
{
  std::ofstream file(filename.c_str());
  file << T.format(CSVFormat);
}

void select_cut_type(
  const int & cut_idx,
  std::string & cut_type)
{
  switch (cut_idx)
  {
    case 0:
      cut_type = "planar";
      break;
    case 1:
      cut_type = "parabolic";
      break;
    case 2:
      cut_type = "sine";
      break;
    case 3:
      cut_type = "square";
      break;
    case 4:
      cut_type = "pulse";
      break;
  }
}

int main(int argc, char * argv[])
{
  cout << "Reading the input mesh...\n";
  std::string mesh_file = argv[1];
  Eigen::MatrixXd mesh_v, tmp_mesh_v;
  Eigen::MatrixXi mesh_f;
  igl::read_triangle_mesh(mesh_file, mesh_v, mesh_f);

  std::cout << "Recentering the input mesh...\n";
  recenter_mesh(mesh_v, mesh_f);

  // Get the max bounding box length and resize the mesh
  double max_bbox_length = (mesh_v.colwise().maxCoeff() - mesh_v.colwise().minCoeff()).maxCoeff();
  mesh_v = mesh_v / max_bbox_length;

  // Update the max bounding box length (should be 1)
  max_bbox_length = (mesh_v.colwise().maxCoeff() - mesh_v.colwise().minCoeff()).maxCoeff();

  // Compute the volume of the resized mesh
  double mesh_volume;
  get_mesh_volume(mesh_v, mesh_f, mesh_volume);
  
  // Get the bounding box diagonal length
  const double bbd = (mesh_v.colwise().maxCoeff() - mesh_v.colwise().minCoeff()).norm();

  // loop through each cut type
  int num_instances = 10;
  for(int cut_idx=0; cut_idx<5; cut_idx++)
  {
    std::string cut_type;
    select_cut_type(cut_idx, cut_type);

    std::cout << "Generating a hyperplane using a " << cut_type << " function...\n";
    Eigen::MatrixXd cut_surface_v;
    Eigen::MatrixXi cut_surface_f;
    int cut_surface_resolution = 100;   
    generate_hyperplane(cut_surface_resolution, cut_type, cut_surface_v, cut_surface_f);
    cut_surface_v = cut_surface_v * (3*bbd); // Scale bigger than object

    int instance_idx = 0;
    while(instance_idx < num_instances)
    {
      const auto update = [&](const std::string & cut_type)
      {     
        Eigen::MatrixXd L_mesh_v, R_mesh_v;
        Eigen::MatrixXi L_mesh_f, R_mesh_f;
        Eigen::MatrixXi L_birth_place, R_birth_place; // keep track of the birth place of each mesh surface

        int L_num_of_components = 0, R_num_of_components = 0;
        double L_mesh_volume = mesh_volume, R_mesh_volume = mesh_volume;
        double volume_ratio = 0.25;
        int counter = 0;
        while(L_num_of_components != 1 || R_num_of_components != 1 || L_mesh_volume < volume_ratio * mesh_volume || R_mesh_volume < volume_ratio * mesh_volume)
        {
          std::cout << "Hyperplane and object intersection checking...\n";
          bool first_only = true;
          bool intersect = false;
          while(!intersect)
          {
            std::cout << "Randomly rotating and translating the object mesh...\n";
            tmp_mesh_v = (igl::random_quaternion<double>().toRotationMatrix() * mesh_v.transpose()).transpose();
            tmp_mesh_v.rowwise() += 0.1 * Eigen::RowVector3d::Random();
            Eigen::MatrixXi IF;
            intersect = igl::copyleft::cgal::intersect_other(tmp_mesh_v, mesh_f, cut_surface_v, cut_surface_f, first_only, IF);
          }

          std::cout << "Trimming the input mesh...\n";
          igl::MeshBooleanType mesh_intersect = igl::MESH_BOOLEAN_TYPE_INTERSECT;
          igl::copyleft::cgal::mesh_boolean(tmp_mesh_v, mesh_f, cut_surface_v, cut_surface_f, mesh_intersect, L_mesh_v, L_mesh_f, L_birth_place);
          igl::MeshBooleanType mesh_minus = igl::MESH_BOOLEAN_TYPE_MINUS;
          igl::copyleft::cgal::mesh_boolean(tmp_mesh_v, mesh_f, cut_surface_v, cut_surface_f, mesh_minus, R_mesh_v, R_mesh_f, R_birth_place);

          std::cout << "Computing the number of components...\n";
          compute_number_of_components(L_mesh_f, L_num_of_components);
          compute_number_of_components(R_mesh_f, R_num_of_components);
          std::cout << "L num of components: " << L_num_of_components << "\n";
          std::cout << "R num of components: " << R_num_of_components << "\n";

          std::cout << "Checking the volumes of the resulting meshes...\n";
          get_mesh_volume(L_mesh_v, L_mesh_f, L_mesh_volume);
          get_mesh_volume(R_mesh_v, R_mesh_f, R_mesh_volume);
          counter += 1;
          if(counter == 5)
          {
            break;
          }
        } // while loop for computing number of components

        if(counter == 5)
          return;

        std::cout << "Recentering meshes...\n";
        Eigen::RowVector3d mesh_c; // Centroid of tmp_mesh_v
        igl::centroid(tmp_mesh_v, mesh_f, mesh_c);
        L_mesh_v.rowwise() -= mesh_c;
        R_mesh_v.rowwise() -= mesh_c;

        std::cout << "Sampling point clouds...\n";
        int num_points = 1024;
        Eigen::MatrixXd L_pc, R_pc;
        Eigen::MatrixXd L_normal, R_normal;
        sample_pc_with_blue_noise(num_points, L_mesh_v, L_mesh_f, L_pc, L_normal);
        sample_pc_with_blue_noise(num_points, R_mesh_v, R_mesh_f, R_pc, R_normal);

        // Save root dir
        std::string save_root_dir = argv[2];
        if(!std::filesystem::is_directory(save_root_dir))
          std::filesystem::create_directory(save_root_dir);

        // category root dir
        std::string category_name = argv[3];
        std::string category_root_dir = save_root_dir + "/" + category_name + "/";
        if(!std::filesystem::is_directory(category_root_dir))
          std::filesystem::create_directory(category_root_dir);

        // object root dir
        std::string object_name = argv[4];
        std::string object_root_dir = category_root_dir + object_name + "/";
        if(!std::filesystem::is_directory(object_root_dir))
          std::filesystem::create_directory(object_root_dir);

        // solid root dir
        std::string solid_root_dir = object_root_dir + "solid/";
        if(!std::filesystem::is_directory(solid_root_dir))
          std::filesystem::create_directory(solid_root_dir);

        // cut root dir
        std::string cut_root_dir = solid_root_dir + cut_type + "/";
        if(!std::filesystem::is_directory(cut_root_dir))
          std::filesystem::create_directory(cut_root_dir);

        // instance root dir
        std::string instance_root_dir = cut_root_dir + std::to_string(instance_idx) + "/";
        if(!std::filesystem::is_directory(instance_root_dir))
          std::filesystem::create_directory(instance_root_dir);

        std::cout << "Saving results in " << instance_root_dir << "...\n";
        std::string L_mesh_file = instance_root_dir + "partA.obj";
        std::string R_mesh_file = instance_root_dir + "partB.obj";
        igl::write_triangle_mesh(L_mesh_file, L_mesh_v, L_mesh_f);
        igl::write_triangle_mesh(R_mesh_file, R_mesh_v, R_mesh_f);

        std::cout << "Saving point clouds...\n";
        std::string L_pc_file = instance_root_dir + "partA-pc.csv";
        std::string R_pc_file = instance_root_dir + "partB-pc.csv";
        write_matrix_to_csv(L_pc_file, L_pc);
        write_matrix_to_csv(R_pc_file, R_pc);

        std::cout << "Saving normals...\n";
        std::string L_normal_file = instance_root_dir + "partA-normal.csv";
        std::string R_normal_file = instance_root_dir + "partB-normal.csv";
        write_matrix_to_csv(L_normal_file, L_normal);
        write_matrix_to_csv(R_normal_file, R_normal);

        instance_idx += 1;

      }; // end of the update function

      update(cut_type);

    } // end of for loop for num of instances

  } // end of for loop for cut type

  return 0;

} // end of main
