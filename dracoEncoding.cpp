#include <string>
#include <fstream>
#include <iostream>
#include <cstdlib>
#include <vector>
#include <sstream>
#include <array>
// https://stackoverflow.com/questions/62313136/how-to-include-the-draco-dec-library
// g++ dracoEncoding.cpp -I /groups/scicompsoft/home/ackermand/local/include/ -L /groups/scicompsoft/home/ackermand/local/lib/ -ldraco -o convertToQuantizedDracoMesh

using Vector3d = std::array<float,3>;//std::array<int64_t, 3>;
using VertexCoord = std::uint32_t;

struct MinAndMaxPositions{
	int minimum[3];
	int maximum[3];
};



struct Mesh{
  std::vector<std::vector<float>> vertices;
  std::vector<std::vector<int>> faces;

  Vector3d fragment_origin;
  Vector3d fragment_shape;
  Vector3d input_origin;
};

class MeshSubdivider{
	//"""A class to read, write and split meshes"""
	const std::string *filename, *output_dir;
	int level, maximum_level;
  int *minimum, *maximum;
  	
  public:
   	 MeshSubdivider (const std::string *, const std::string *, int, int, MinAndMaxPositions);
};
std::map<Vector3,int> mymap;
Mesh ReadObjFile(const std::string * filename){
  std::ifstream myfile (*filename);
  std::string line;

  std::vector<std::vector<float>> vertices;
  std::vector<std::vector<int>> faces;

  std::string v;
  float x,y,z;
  while(!myfile.eof())
  {
      getline (myfile,line);
      if (line[0] == 'v')
      {
          std::vector<float> vertex;
          std::istringstream iss( line );
          iss>>v>>x>>y>>z;
          vertex.push_back(x);
          vertex.push_back(y);
          vertex.push_back(z);

          vertices.push_back(vertex);
      }
      else if(line[0] == 'f'){
          std::vector<int> face;
          std::istringstream iss( line );
          iss>>v>>x>>y>>z;
          face.push_back(x);
          face.push_back(y);
          face.push_back(z);

          faces.push_back(face);
      }
  }

  Mesh mesh = {vertices,faces};
  std::cout<<vertices.size();
  return mesh;
}

MeshSubdivider::MeshSubdivider(const std::string * filename, const std::string * output_dir, int level, int maximum_level, MinAndMaxPositions minAndMaxPositions){
  this->filename = filename;
  this->output_dir = output_dir;
  this->level = level;
  this->maximum_level = level;
  this->minimum = minAndMaxPositions.minimum;
  this->maximum = minAndMaxPositions.maximum;

  Mesh original_mesh = ReadObjFile(this->filename);
};

/*
void SubdivideMesh(std::string & directory, Vector3d & fragment_shape, int * fragment_position){
  Mesh mesh = ReadObjFile(directory,fragment_shape, fragment_position);
  int num_quantization_bits = 10;

  Quantizer quantizer(mesh.fragment_origin, mesh.fragment_shape,
                                     mesh.input_origin, num_quantization_bits);

  draco::TriangleSoupMeshBuilder mb;
  using VertexCoord = std::uint32_t;

  int num_triangles = mesh.vertices.size();
  if(num_triangles>0){
    mb.Start(num_triangles);
    int position_att_id = mb.AddAttribute(draco::GeometryAttribute::POSITION, 3, draco::DT_UINT32);
    int face_index = 0;
    for (std::vector<int> face : mesh.faces){

        std::array<std::array<VertexCoord, 3>, 3> points;
        for (int i = 0; i < 3; ++i) {
          int face_index = face.at(i) -1;
          points[i] = quantizer(mesh.vertices.at( face_index ));
        }

        mb.SetAttributeValuesForFace(position_att_id, draco::FaceIndex(face_index),
                                       &points[0], &points[1], &points[2]);
        face_index++;
    }

    std::unique_ptr<draco::Mesh> draco_mesh = mb.Finalize();  
    draco::EncoderBuffer eb;
    draco::Encoder encoder;
    encoder.SetEncodingMethod(draco::MESH_EDGEBREAKER_ENCODING);
    encoder.SetAttributePredictionScheme(draco::GeometryAttribute::POSITION,
                                         draco::MESH_PREDICTION_PARALLELOGRAM);
    auto draco_status = encoder.EncodeMeshToBuffer(*draco_mesh, &eb);
    
    std::string output_filename=directory;
    for(int d=0; d<3; d++){
      output_filename+= std::to_string(fragment_position[d]);
    }
    output_filename+=".drc";
    draco::FileWriterFactory::RegisterWriter(draco::StdioFileWriter::Open);
    draco::WriteBufferToFile(eb.data(), eb.size(), output_filename);
  }

}*/

int main(int argc, char ** argv){

  std::string input_directory = argv[1];
  //Vector3d fragment_shape = {std::strtof(argv[2],NULL),std::strtof(argv[3],NULL),std::strtof(argv[4],NULL)};
  //int fragment_position[3]= {std::atoi(argv[5]),std::atoi(argv[6]),std::atoi(argv[7])};
  ReadObjFile(&input_directory);
    
  return 1;
}