/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <nvimgcodec.h>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace nvimgcodec {

namespace py = pybind11;

class PluginSelectionArg
{
  public:
    PluginSelectionArg(const py::object& value, const char* argument_name)
        : argument_name_(argument_name)
    {
        if (value.is_none()) {
            return;
        }
        if (py::isinstance<py::str>(value)) {
            throw py::type_error(argument_name_ + " must be a sequence of strings or a mapping");
        }

        py::object mapping_type = py::module_::import("collections.abc").attr("Mapping");
        if (py::isinstance(value, mapping_type)) {
            py::object items = value.attr("items")();
            for (py::handle item : items) {
                py::tuple pair = py::reinterpret_borrow<py::tuple>(item);
                if (!py::isinstance<py::str>(pair[0])) {
                    throw py::type_error(argument_name_ + " mapping keys must be strings");
                }
                std::string codec_name = py::cast<std::string>(pair[0]);
                if (codec_name.empty()) {
                    throw py::value_error(argument_name_ + " contains an empty codec name");
                }
                if (codec_name.find('\0') != std::string::npos) {
                    throw py::value_error(argument_name_ + " codec names must not contain null characters");
                }
                codec_names_.emplace_back(std::move(codec_name));
                plugin_ids_.push_back(parseIds(pair[1]));
            }
            if (codec_names_.empty()) {
                throw py::value_error(argument_name_ + " must not be empty");
            }
        } else {
            codec_names_.emplace_back(std::nullopt);
            plugin_ids_.push_back(parseIds(value));
        }

        id_pointers_.resize(plugin_ids_.size());
        selections_.resize(plugin_ids_.size());
        for (size_t selection_idx = 0; selection_idx < plugin_ids_.size(); ++selection_idx) {
            auto& pointers = id_pointers_[selection_idx];
            pointers.reserve(plugin_ids_[selection_idx].size());
            for (const std::string& plugin_id : plugin_ids_[selection_idx]) {
                pointers.push_back(plugin_id.c_str());
            }
            selections_[selection_idx] = nvimgcodecPluginSelection_t{NVIMGCODEC_STRUCTURE_TYPE_PLUGIN_SELECTION,
                sizeof(nvimgcodecPluginSelection_t), nullptr, codec_names_[selection_idx] ? codec_names_[selection_idx]->c_str() : nullptr,
                static_cast<int>(pointers.size()), pointers.data()};
        }
    }

    void apply(nvimgcodecExecutionParams_t* exec_params) const
    {
        exec_params->num_selections = static_cast<int>(selections_.size());
        exec_params->selections = selections_.empty() ? nullptr : selections_.data();
    }

    bool specified() const { return !selections_.empty(); }

    void validateDecoderIds(nvimgcodecInstance_t instance) const
    {
        validateIds<nvimgcodecDecoderDesc_t>(
            instance, [](const nvimgcodecFrameworkDesc_t* framework, const nvimgcodecDecoderDesc_t** descs, int* count) {
                return framework->getDecoderDescs(framework->instance, descs, count, nullptr, NVIMGCODEC_BACKEND_KIND_ANY);
            });
    }

    void validateEncoderIds(nvimgcodecInstance_t instance) const
    {
        validateIds<nvimgcodecEncoderDesc_t>(
            instance, [](const nvimgcodecFrameworkDesc_t* framework, const nvimgcodecEncoderDesc_t** descs, int* count) {
                return framework->getEncoderDescs(framework->instance, descs, count, nullptr, NVIMGCODEC_BACKEND_KIND_ANY);
            });
    }

  private:
    static void checkStatus(nvimgcodecStatus_t status)
    {
        if (status != NVIMGCODEC_STATUS_SUCCESS) {
            throw std::runtime_error("nvImageCodec failure while validating plugin selection: " + std::to_string(status));
        }
    }

    template <typename Desc, typename Query>
    void validateIds(nvimgcodecInstance_t instance, Query&& query) const
    {
        if (!specified()) {
            return;
        }

        const nvimgcodecFrameworkDesc_t* framework = nullptr;
        checkStatus(nvimgcodecGetFrameworkDesc(instance, &framework));
        int count = 0;
        checkStatus(query(framework, nullptr, &count));
        std::vector<const Desc*> descs(count);
        if (count != 0) {
            checkStatus(query(framework, descs.data(), &count));
            descs.resize(count);
        }

        std::unordered_map<std::string, std::vector<std::string>> codecs_by_id;
        std::unordered_map<std::string, std::unordered_map<std::string, size_t>> id_counts_by_codec;
        for (const Desc* desc : descs) {
            if (desc == nullptr || desc->id == nullptr || desc->codec == nullptr) {
                throw std::runtime_error("Registered plugin description is incomplete");
            }
            codecs_by_id[desc->id].emplace_back(desc->codec);
            ++id_counts_by_codec[desc->codec][desc->id];
        }

        for (size_t selection_idx = 0; selection_idx < plugin_ids_.size(); ++selection_idx) {
            for (const std::string& plugin_id : plugin_ids_[selection_idx]) {
                if (!codec_names_[selection_idx]) {
                    auto matches = codecs_by_id.find(plugin_id);
                    if (matches == codecs_by_id.end()) {
                        throw py::value_error(argument_name_ + " contains unknown plugin identifier '" + plugin_id + "'");
                    }
                    if (matches->second.size() != 1) {
                        throw py::value_error(
                            argument_name_ + " contains ambiguous plugin identifier '" + plugin_id + "'; use a per-codec mapping");
                    }
                } else {
                    const std::string& codec_name = *codec_names_[selection_idx];
                    auto codec = id_counts_by_codec.find(codec_name);
                    auto plugin = codec == id_counts_by_codec.end() ? nullptr : &codec->second;
                    if (plugin == nullptr || plugin->count(plugin_id) != 1 || plugin->at(plugin_id) != 1) {
                        throw py::value_error(
                            argument_name_ + " plugin identifier '" + plugin_id + "' is not registered for codec '" + codec_name + "'");
                    }
                }
            }
        }
    }

    std::vector<std::string> parseIds(py::handle value) const
    {
        py::object sequence_type = py::module_::import("collections.abc").attr("Sequence");
        if (py::isinstance<py::str>(value) || !py::isinstance(value, sequence_type)) {
            throw py::type_error(argument_name_ + " values must be sequences of strings");
        }
        std::vector<std::string> ids;
        for (py::handle item : py::reinterpret_borrow<py::iterable>(value)) {
            if (!py::isinstance<py::str>(item)) {
                throw py::type_error(argument_name_ + " must contain only strings");
            }
            ids.push_back(py::cast<std::string>(item));
        }
        if (ids.empty()) {
            throw py::value_error(argument_name_ + " selections must not be empty");
        }

        std::unordered_set<std::string> unique_ids;
        for (const std::string& plugin_id : ids) {
            if (plugin_id.empty()) {
                throw py::value_error(argument_name_ + " contains an empty plugin identifier");
            }
            if (plugin_id.find('\0') != std::string::npos) {
                throw py::value_error(argument_name_ + " plugin identifiers must not contain null characters");
            }
            if (!unique_ids.insert(plugin_id).second) {
                throw py::value_error(argument_name_ + " contains duplicate plugin identifier '" + plugin_id + "'");
            }
        }
        return ids;
    }

    std::string argument_name_;
    std::vector<std::optional<std::string>> codec_names_;
    std::vector<std::vector<std::string>> plugin_ids_;
    std::vector<std::vector<const char*>> id_pointers_;
    std::vector<nvimgcodecPluginSelection_t> selections_;
};

} // namespace nvimgcodec
