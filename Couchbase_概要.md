# Couchbase

## 简介

Couchbase是由MemBase和CouchDB合并而成的NoSQL数据库管理系统，结合了CouchDB的可靠性和Memcached的高性能。
它可以用作管理缓存层、key-value存储和文档型数据库。

## Couchbase Server的功能

Couchbase Server可以作为以下三种角色：

- 管理缓存层
- key-value存储
- 文档型数据库

## 可伸缩性(Scalability)

Couchbase的可伸缩性特点包括：

- 一键式添加和删除节点，无需停机。
- 自动分片（Auto-sharding），实现跨集群的自动负载平衡，避免热点问题。
- 跨数据中心复制功能，增强了Couchbase的跨地域可扩展性。

## 自动分片集群技术

Couchbase在添加或删除服务器时，数据会自动重新分配，实现不停机的重新平衡。自动分片确保数据在集群中均匀分布，提高数据局部性，减少网络流量，提升响应时间。

## 高性能、高吞吐量和低延迟

Couchbase为交互式Web应用程序提供高吞吐量和低延迟，即使在高负荷下也能保持一致的毫秒级延迟。
Couchbase内置了基于memcache的缓存技术，优化磁盘I/O和CPU，提高响应时间和吞吐量。

## 集群高可用性

Couchbase在集群情况下保持高可用性，具有以下优势：

- 跨数据中心复制
- 即使在整个数据中心失败的情况下也提供高可用性
- 软件和硬件升级可以在不关闭服务器的情况下在线完成
- 维护操作如压缩也可以在服务在线的情况下完成

## 跨数据中心复制(XDCR)

Couchbase通过XDCR在不同数据中心的集群间复制数据，提供数据局部性，减少网络延迟。XDCR可以单向或双向复制数据，允许在不同数据中心间添加和读取数据。

## 数据局部性(Data Locality)

数据局部性是指Couchbase服务器与客户端的接近程度。Couchbase通过跨数据中心复制，使得数据可以在最靠近客户端的数据中心服务，减少网络延迟。

## 管理和监控图形用户界面

Couchbase提供了管理和监控的图形用户界面，方便用户进行操作和监控。

## 处理节点故障和数据修复

Couchbase使用复制和分片来处理节点故障和数据修复。

在复制方面，Couchbase使用数据复制来确保在节点故障时不会丢失数据。
每个桶都有一个可配置的副本数量，该数量指定了要将数据复制到多少个节点。当数据写入主节点时，它会自动复制到副本节点。
当主节点发生故障时，Couchbase会自动将其中一个副本提升为新的主节点，以确保数据可用性。
一旦原始主节点恢复，它将成为一个副本节点，并自动同步数据。

在分片方面，Couchbase使用数据分片将数据分布在多个节点上。
每个桶都可以配置多个分片，Couchbase使用一致性哈希算法来分配数据到各个分片。
当节点故障时，Couchbase会自动将该节点上的数据分配到其他节点上的可用分片。
一旦节点恢复，Couchbase会自动将数据分配回该节点。

数据修复是通过使用副本节点来实现的。
当主节点发生故障时，Couchbase会自动将其中一个副本提升为新的主节点，并使用其他副本节点来修复数据。
如果有任何数据损坏或缺失，Couchbase会自动从其他副本节点中获取数据并进行修复。
一旦数据修复完成，Couchbase会自动将节点重新添加到集群中，以便它可以成为一个新的副本节点

## 处理并发访问和锁定

- CAS操作
  Couchbase使用CAS操作来实现乐观并发控制。当多个客户端同时尝试修改同一个文档时，Couchbase会检查文档的CAS值，如果CAS值相同，则允许更新文档。如果CAS值不同，则表示文档已被其他客户端更新，需要重新尝试更新。

- Bucket级别锁定
  Couchbase使用桶级别锁定来避免多个客户端同时修改同一个桶中的文档。当一个客户端更新一个文档时，Couchbase会锁定整个桶，直到更新完成为止。

- 乐观锁定
  Couchbase使用乐观锁定来避免多个客户端同时修改同一个文档。当一个客户端更新一个文档时，Couchbase会检查文档的CAS值，如果CAS值相同，则允许更新文档。如果CAS值不同，则表示文档已被其他客户端更新，需要重新尝试更新。

- 悲观锁定
  Couchbase支持悲观锁定来避免多个客户端同时修改同一个文档。当一个客户端需要修改一个文档时，可以使用悲观锁定锁定文档，直到更新完成为止。其他客户端需要等待锁定释放后才能修改文档。

## 应用场景

- 会话存储和缓存
  Couchbase可以作为会话存储或缓存，以提高应用程序性能和可伸缩性。它可以存储和管理大量的会话数据，并提供快速的读写操作。

- 用户和设备数据存储
  Couchbase可以用于存储和管理用户和设备数据，例如用户配置文件、设备信息、设备状态等。它可以处理大量的实时数据，并提供可伸缩性和高性能。

- 实时数据处理
  Couchbase可以处理实时数据，例如IoT设备生成的数据、移动应用程序生成的数据等。它可以快速地读写数据，并提供实时的数据查询和分析功能。

- 分布式缓存
  Couchbase可以用于构建分布式缓存，以提高应用程序性能和可伸缩性。它可以存储和管理大量的缓存数据，并提供快速的读写操作。

- 大数据存储和处理
  Couchbase可以用于存储和处理大数据，例如日志数据、事务数据等。它可以处理大量的实时数据，并提供可伸缩性和高性能。

Couchbase Lite能够在任何网络环境下，确保移动应用对本地数据库的响应性，即使在离线状态下，也不会妨碍数据库执行常规的增加、删除、修改等操作。
此外，一旦设备恢复网络连接，Couchbase Lite将通过Couchbase Sync Gateway与服务器进行数据同步。
Couchbase Sync Gateway作为安全的数据访问和同步的Web网关，其核心功能是将数据同步至Couchbase服务器以及多个Couchbase Lite实例，
确保数据的一致性和实时性。
这一机制不仅增强了移动应用的用户体验，也为数据的安全性和可靠性提供了有力保障。

## Couchbash C++ 

[Distributed Transactions from the C++ SDK](https://docs.couchbase.com/cxx-txns/current/distributed-acid-transactions-from-the-sdk.html)

安装:

- Couchbase 服务器
- Boost
- CMake
- C++编译器
- couchbase-transactions-cxx
- libssl 1.1
- [transactions-cxx](https://github.com/couchbase/couchbase-transactions-cxx.git)

安装:

- CMake
- C++编译器
- [CPM.cmake](https://github.com/cpm-cmake/CPM.cmake#comparison-to-pure-fetchcontent--externalproject)
- [NASM编译器](https://www.nasm.us/pub/nasm/releasebuilds/2.16/win64/)

在项目中添加:

```CMake
CPMAddPackage(
NAME
couchbase_cxx_client
GIT_TAG
1.0.3
VERSION
1.0.3
GITHUB_REPOSITORY
"couchbase/couchbase-cxx-client"
OPTIONS
"COUCHBASE_CXX_CLIENT_STATIC_BORINGSSL ON"
"COUCHBASE_CXX_CLIENT_BUILD_SHARED OFF"
"COUCHBASE_CXX_CLIENT_BUILD_STATIC ON")
```

头文件和库文件需要手动添加:

```cmake
target_link_libraries(database PRIVATE fmt::fmt)
target_link_libraries(database PRIVATE couchbase_cxx_client::couchbase_cxx_client_static)
target_link_libraries(database PRIVATE taocpp::json)

target_include_directories(database
        PUBLIC
        $<BUILD_INTERFACE:${couchbase_cxx_client_SOURCE_DIR}>
)
```
