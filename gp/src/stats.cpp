#include <vector>
#include <cmath>
#include <cstddef>


float mean(const std::vector<float>& scores)
{
  size_t N = scores.size();
  float m = 0.0f;
  for (size_t i=0; i<N; ++i) {
    m += scores[i];
  }
  return m/static_cast<float>(N);
}


float stdev(const std::vector<float>& scores)
{
  size_t N = scores.size();
  float m = mean(scores);

  float sd = 0.0f;
  for (size_t i=0; i<N; ++i) {
    sd += (scores[i]-m)*(scores[i]-m);
  }
  sd = sd/static_cast<float>(N);
  return std::sqrt(sd);
}
