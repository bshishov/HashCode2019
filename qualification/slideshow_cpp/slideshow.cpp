// slideshow.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <map>
#include <list>
#include <omp.h>
#include <chrono>
#include <functional>

#include <Eigen/Sparse>

struct ScopedTimer
{
	std::chrono::high_resolution_clock::time_point t0_;
	std::string title_;

	ScopedTimer(std::string title)
		: t0_(std::chrono::high_resolution_clock::now())
		, title_(title)
	{}

	~ScopedTimer()
	{
		auto t1 = std::chrono::high_resolution_clock::now();
		auto nanos = std::chrono::duration_cast<std::chrono::seconds>(t1 - t0_).count();

		std::cout << title_ << " took " << nanos << "s to complete" << std::endl;
	}
};

enum OrientationMode
{
	Horizontal,
	Vertical
};

struct Stats
{
	uint64_t min_tags_count = -1;
	uint64_t max_tags_count = 0;
	uint64_t avg_tags_count = 0;
};

struct Photo
{
	uint32_t idx;
	OrientationMode orientation;
	std::vector<uint32_t> tags;
};

bool photoLess(const Photo& p1, const Photo& p2)
{
	return p1.tags.size() < p2.tags.size();
}

void BuildAdjacencyMatrix(std::vector<Photo> &photos, Stats &s, Eigen::SparseMatrix<uint8_t> &mat)
{
	const uint64_t photos_count = photos.size();

	std::vector<uint8_t> intersections_per_photo(photos_count, 0);
	std::vector<uint64_t> intersected_tags_count(s.max_tags_count, 0);

	mat.reserve(photos.size() * 4);

#pragma omp parallel for shared(photos, mat) num_threads(8)
	for (int i = 0; i < photos_count - 1; ++i)
	{
		const auto& curent_photo = photos[i];
		for (int k = i + 1; k < photos_count; ++k)
		{
			const auto& matching_photo = photos[k];
			std::vector<uint32_t> tmp;
			std::set_intersection(curent_photo.tags.begin(), curent_photo.tags.end(),
								  matching_photo.tags.begin(), matching_photo.tags.end(),
								  std::back_inserter(tmp));

			const uint64_t intersected_tags = tmp.size();
			if (intersected_tags == 0)
				continue;

			++intersected_tags_count[intersected_tags];
			++intersections_per_photo[i];
			++intersections_per_photo[k];

			uint64_t score = std::min(std::min(intersected_tags, curent_photo.tags.size()), matching_photo.tags.size());

#pragma omp critical
			{
				mat.insert(i, k) = score;
				mat.insert(k, i) = score;
			}
		}
	}
	mat.makeCompressed();

	auto minmaxIt = std::minmax_element(intersections_per_photo.begin(), intersections_per_photo.end());
	std::cout << "min intersections: " << *minmaxIt.first << " max intersections: " << *minmaxIt.second << std::endl;
}

template <typename ResultIt>
bool BuildRoute(Photo current_photo, std::list<Photo> &photos, ResultIt resultIterator, const bool untillTheEnd = true)
{
	uint64_t length_of_slides = 0;
	while (true)
	{

		std::vector<uint32_t> tmp;
		auto nextPhotoIt = std::find_if(photos.begin(), photos.end(), [&](const Photo &p) {
			tmp.clear();
			std::set_intersection(current_photo.tags.begin(), current_photo.tags.end(),
								  p.tags.begin(), p.tags.end(),
								  std::back_inserter(tmp));

			const uint64_t intersected_tags = tmp.size();
			return intersected_tags > 0;
		});

		if (nextPhotoIt != photos.end())
		{
			resultIterator = nextPhotoIt->idx;
			current_photo = *nextPhotoIt;
			++length_of_slides;

			photos.erase(nextPhotoIt);

			if(!untillTheEnd)
				break;
		}
		else
		{
			std::cout << "Length of path = " << length_of_slides << std::endl;
			break;
		}
	}

	return length_of_slides;
}

std::list<uint64_t> Solve(Stats &s, std::vector<Photo> &photos)
{
	Eigen::SparseMatrix<uint8_t> mat(photos.size(), photos.size());

	std::sort(photos.begin(), photos.end(), photoLess);
	const uint64_t photos_count = photos.size();


	uint64_t current_index = 0;
	std::vector<uint32_t> tmp;
	std::list<Photo> tmp_photos(photos.begin(), photos.end());

	std::list<uint64_t> final_results;
	uint64_t length_of_slides = 1;

	bool reverse_order = false;
	while (!tmp_photos.empty())
	{
		auto current_it = tmp_photos.begin();
		auto current_photo = *current_it;
		tmp_photos.erase(current_it);

		std::list<uint64_t> current_results;
		current_results.push_back(current_photo.idx);

		bool anyFind = false;
		do
		{
			anyFind = false;
			anyFind |= BuildRoute(current_photo, tmp_photos, std::back_inserter(current_results), true);
			anyFind |= BuildRoute(current_photo, tmp_photos, std::front_inserter(current_results), true);

		} while (anyFind);

		final_results.splice(final_results.end(), current_results);

	}

	return std::move(final_results);
}

int main(int argc, char **argv)
{
	ScopedTimer overall("Whole Process");

	if (argc != 3)
		std::cout << "Invalid parameter!\n";

	std::vector<Photo> photos;

	Stats s;
	{
		ScopedTimer preprocess("Preprocess");

		std::ifstream inputFileStream(argv[1]);
		int count;
		inputFileStream >> count;
		inputFileStream.ignore(1, '\n');

		std::string line;
		photos.reserve(count);

		std::unordered_map<std::string, uint32_t> tag_ids;
		uint32_t current_tag_id = 0;
		for (int i = 0; i < count; i++)
		{
			std::getline(inputFileStream, line);

			std::istringstream iss(line);
			char orientation = 0;
			iss >> orientation;
			int tag_count = 0;
			iss >> tag_count;

			Photo p = {};
			p.orientation = orientation == 'V' ? OrientationMode::Vertical : OrientationMode::Horizontal;
			p.idx = i;
			p.tags.reserve(tag_count);

			for (int j = 0; j < tag_count; ++j)
			{
				std::string tag;
				iss >> tag;

				auto it = tag_ids.find(tag);
				if (it == tag_ids.end())
				{
					it = tag_ids.insert(std::make_pair(tag, current_tag_id++)).first;
				}
				p.tags.push_back(it->second);
			}
			sort(p.tags.begin(), p.tags.end(), std::less<uint32_t>());

			if (tag_count < s.min_tags_count)
				s.min_tags_count = tag_count;
			if (tag_count > s.max_tags_count)
				s.max_tags_count = tag_count;
			s.avg_tags_count += tag_count;

			photos.emplace_back(std::move(p));
		}
		s.avg_tags_count /= photos.size();
	}

	std::cout << "min_tags: " << s.min_tags_count << "max_tags: " << s.max_tags_count << "avg_tags: " << s.avg_tags_count << std::endl;
	std::list<uint64_t> res;
	{
		ScopedTimer solve("Solve");
		res = Solve(s, photos);
	}

	ScopedTimer write_res("Write results");

	std::ofstream output_file(argv[2]);
	output_file << res.size() << "\n";
	std::ostream_iterator<uint64_t> output_iterator(output_file, "\n");
	std::copy(res.begin(), res.end(), output_iterator);
}