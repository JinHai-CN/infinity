//
// Created by jinhai on 23-10-14.
//

module;

#include <boost/bind.hpp>
#include <thread>

module db_server;
import infinity_context;
import stl;
import boost;
import third_party;
import infinity_exception;
import infinity_assert;
import connection;

namespace infinity {

void DBServer::Init(const StartupParameter &parameter) { config_path_ = std::move(parameter.config_path); }

void DBServer::Run() {
    if (initialized) {
        return;
    }

    initialized = true;

    InfinityContext::instance().Init(config_path_);
    u16 pg_port = InfinityContext::instance().config()->pg_port();
    const String &listen_address_ref = InfinityContext::instance().config()->listen_address();

    BoostErrorCode error;
    AsioIpAddr address = asio_make_address(listen_address_ref, error);
    if (error) {
        Error<NetworkException>(Format("Not a valid IPv4 address: {}", listen_address_ref), __FILE_NAME__, __LINE__);
    }

    acceptor_ptr_ = MakeUnique<AsioAcceptor>(io_service_, AsioEndPoint(address, pg_port));
    CreateConnection();

    if (config_path_) {
        Printf("Startup database server, at: {} and port: {}, config: {}\n", listen_address_ref, pg_port, *config_path_);
    } else {
        Printf("Startup database server, at: {} and port: {}\n", listen_address_ref, pg_port);
    }

    InfinityContext::instance().config()->PrintAll();

    Printf("Run 'psql -h {} -p {}' to connect to the server.\n", listen_address_ref, pg_port);

    io_service_.run();
}

void DBServer::Shutdown() {
    Printf("Shutdown infinity server ...\n");
    while (running_connection_count_ > 0) {
        // Running connection exists.
        std::this_thread::yield();
    }

    io_service_.stop();
    initialized = false;
    acceptor_ptr_->close();
    infinity::InfinityContext::instance().UnInit();
    Printf("Shutdown infinity server successfully\n");
}

void DBServer::CreateConnection() {
    SharedPtr<Connection> connection_ptr = MakeShared<Connection>(io_service_);
    acceptor_ptr_->async_accept(*(connection_ptr->socket()), boost::bind(&DBServer::StartConnection, this, connection_ptr));
}

void DBServer::StartConnection(SharedPtr<Connection> &connection) {
    Thread connection_thread([connection = connection, &num_running_connections = this->running_connection_count_]() mutable {
        ++num_running_connections;
        connection->Run();

        // User disconnected
        connection.reset();
        --num_running_connections;
    });

    connection_thread.detach();
    CreateConnection();
}

} // namespace infinity
