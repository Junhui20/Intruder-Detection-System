-- SQLite Database Schema for Intruder Detection System
-- Migrated from MariaDB to SQLite for better performance and simplicity

-- Enable foreign key constraints
PRAGMA foreign_keys = ON;

-- Create the devices table for IP camera management
CREATE TABLE IF NOT EXISTS devices (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ip_address TEXT NOT NULL,
    port INTEGER NOT NULL,
    use_https BOOLEAN DEFAULT 0,
    end_with_video BOOLEAN DEFAULT 0,
    status TEXT DEFAULT 'active' CHECK(status IN ('active', 'inactive')),
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Create index for faster device queries
CREATE INDEX IF NOT EXISTS idx_devices_status ON devices(status);
CREATE INDEX IF NOT EXISTS idx_devices_ip_port ON devices(ip_address, port);

-- Whitelist table for authorized individuals and recognized animals
CREATE TABLE IF NOT EXISTS whitelist (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    entity_type TEXT NOT NULL CHECK(entity_type IN ('human', 'animal')),
    familiar TEXT DEFAULT 'familiar' CHECK(familiar IN ('familiar', 'unfamiliar')),
    color TEXT,
    coco_class_id INTEGER,
    image_path TEXT NOT NULL,
    confidence_threshold REAL DEFAULT 0.6,
    -- Individual pet identification fields
    pet_breed TEXT,                    -- e.g., 'golden_retriever', 'persian_cat'
    individual_id TEXT,                -- e.g., 'jacky', 'fluffy'
    face_encodings BLOB,               -- Stored face encodings for pet identification
    multiple_photos TEXT,              -- JSON array of additional photo paths
    identification_method TEXT DEFAULT 'color' CHECK(identification_method IN ('color', 'face', 'hybrid')),
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for faster whitelist queries
CREATE INDEX IF NOT EXISTS idx_whitelist_entity_type ON whitelist(entity_type);
CREATE INDEX IF NOT EXISTS idx_whitelist_familiar ON whitelist(familiar);
CREATE INDEX IF NOT EXISTS idx_whitelist_coco_class ON whitelist(coco_class_id);

-- Create the notification_settings table for Telegram users
CREATE TABLE IF NOT EXISTS notification_settings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    chat_id INTEGER NOT NULL UNIQUE,
    telegram_username TEXT NOT NULL,
    notify_human_detection BOOLEAN NOT NULL DEFAULT 1,
    notify_animal_detection BOOLEAN NOT NULL DEFAULT 1,
    sendstatus TEXT DEFAULT 'open' CHECK(sendstatus IN ('open', 'close')),
    last_notification DATETIME,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Create index for faster notification queries
CREATE INDEX IF NOT EXISTS idx_notification_chat_id ON notification_settings(chat_id);
CREATE INDEX IF NOT EXISTS idx_notification_status ON notification_settings(sendstatus);

-- Create detection_logs table for tracking detections (new feature)
CREATE TABLE IF NOT EXISTS detection_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    detection_type TEXT NOT NULL CHECK(detection_type IN ('human', 'animal')),
    entity_name TEXT,
    confidence REAL,
    camera_id INTEGER,
    image_path TEXT,
    notification_sent BOOLEAN DEFAULT 0,
    detected_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (camera_id) REFERENCES devices(id)
);

-- Create indexes for detection logs
CREATE INDEX IF NOT EXISTS idx_detection_logs_type ON detection_logs(detection_type);
CREATE INDEX IF NOT EXISTS idx_detection_logs_date ON detection_logs(detected_at);
CREATE INDEX IF NOT EXISTS idx_detection_logs_camera ON detection_logs(camera_id);

-- Create system_metrics table for performance monitoring (new feature)
CREATE TABLE IF NOT EXISTS system_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    metric_type TEXT NOT NULL,
    metric_value REAL NOT NULL,
    unit TEXT,
    recorded_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Create index for metrics queries
CREATE INDEX IF NOT EXISTS idx_metrics_type_date ON system_metrics(metric_type, recorded_at);

-- Create configuration table for system settings (new feature)
CREATE TABLE IF NOT EXISTS system_config (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    config_key TEXT NOT NULL UNIQUE,
    config_value TEXT NOT NULL,
    config_type TEXT DEFAULT 'string' CHECK(config_type IN ('string', 'integer', 'float', 'boolean')),
    description TEXT,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Insert default configuration values
INSERT OR IGNORE INTO system_config (config_key, config_value, config_type, description) VALUES
('yolo_model', 'yolo11n.pt', 'string', 'YOLO11n model file name'),
('yolo_confidence', '0.5', 'float', 'YOLO detection confidence threshold'),
('human_confidence_threshold', '0.6', 'float', 'Configurable confidence threshold for human face recognition'),
('animal_confidence_threshold', '0.6', 'float', 'Configurable confidence threshold for animal detection'),
('pet_identification_threshold', '0.7', 'float', 'Confidence threshold for individual pet identification'),
('multi_face_detection', '1', 'boolean', 'Enable simultaneous multi-face detection'),
('max_faces_per_frame', '10', 'integer', 'Maximum faces to process per frame'),
('unknown_person_timer', '5', 'integer', 'Seconds before unknown person alert'),
('unfamiliar_animal_timer', '5', 'integer', 'Seconds before unfamiliar animal alert'),
('notification_cooldown', '20', 'integer', 'Seconds between notifications'),
('frame_width', '640', 'integer', 'Video frame processing width'),
('frame_height', '480', 'integer', 'Video frame processing height'),
('target_fps', '30', 'integer', 'Target processing FPS'),
('enable_gpu', '1', 'boolean', 'Enable GPU acceleration'),
('enable_performance_monitoring', '1', 'boolean', 'Enable performance metrics collection'),
('pet_identification_method', 'hybrid', 'string', 'Method for pet identification: color, face, or hybrid');

-- Create triggers to update timestamps
CREATE TRIGGER IF NOT EXISTS update_devices_timestamp 
    AFTER UPDATE ON devices
    BEGIN
        UPDATE devices SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
    END;

CREATE TRIGGER IF NOT EXISTS update_whitelist_timestamp 
    AFTER UPDATE ON whitelist
    BEGIN
        UPDATE whitelist SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
    END;

CREATE TRIGGER IF NOT EXISTS update_notification_timestamp 
    AFTER UPDATE ON notification_settings
    BEGIN
        UPDATE notification_settings SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
    END;

CREATE TRIGGER IF NOT EXISTS update_config_timestamp 
    AFTER UPDATE ON system_config
    BEGIN
        UPDATE system_config SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
    END;

-- Create views for easier data access
CREATE VIEW IF NOT EXISTS active_cameras AS
SELECT * FROM devices WHERE status = 'active';

CREATE VIEW IF NOT EXISTS known_humans AS
SELECT * FROM whitelist WHERE entity_type = 'human' AND familiar = 'familiar';

CREATE VIEW IF NOT EXISTS familiar_animals AS
SELECT * FROM whitelist WHERE entity_type = 'animal' AND familiar = 'familiar';

CREATE VIEW IF NOT EXISTS active_telegram_users AS
SELECT * FROM notification_settings WHERE sendstatus = 'open';

-- Sample data insertion (commented out - uncomment for testing)
/*
-- Sample IP camera
INSERT INTO devices (ip_address, port, use_https, end_with_video, status) 
VALUES ('192.168.1.100', 8080, 0, 1, 'active');

-- Sample known person
INSERT INTO whitelist (name, entity_type, familiar, image_path) 
VALUES ('John Doe', 'human', 'familiar', 'faces/john_doe.jpg');

-- Sample familiar animals with individual identification
INSERT INTO whitelist (name, entity_type, familiar, color, coco_class_id, image_path, individual_id, pet_breed, identification_method)
VALUES ('Fluffy', 'animal', 'familiar', 'white', 15, 'animals/fluffy_cat.jpg', 'fluffy', 'persian', 'hybrid');

INSERT INTO whitelist (name, entity_type, familiar, color, coco_class_id, image_path, individual_id, pet_breed, identification_method)
VALUES ('Jacky', 'animal', 'familiar', 'golden', 16, 'animals/jacky_dog.jpg', 'jacky', 'golden_retriever', 'face');

-- Sample Telegram user
INSERT INTO notification_settings (chat_id, telegram_username, notify_human_detection, notify_animal_detection) 
VALUES (123456789, 'john_doe_telegram', 1, 1);
*/
