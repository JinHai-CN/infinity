//
// Created by jinhai on 23-3-18.
//

module;

import stl;

export module like;

namespace infinity {

class NewCatalog;

void RegisterLikeFunction(const UniquePtr<NewCatalog> &catalog_ptr);

void RegisterNotLikeFunction(const UniquePtr<NewCatalog> &catalog_ptr);

} // namespace infinity
