#include <QApplication>
#include <QCheckBox>
#include <QCoreApplication>
#include <QDir>
#include <QDragEnterEvent>
#include <QDropEvent>
#include <QFileDialog>
#include <QFileInfo>
#include <QFont>
#include <QHBoxLayout>
#include <QHeaderView>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QLabel>
#include <QLineEdit>
#include <QMainWindow>
#include <QMimeData>
#include <QProcess>
#include <QProgressBar>
#include <QPushButton>
#include <QSet>
#include <QStandardPaths>
#include <QStatusBar>
#include <QTextEdit>
#include <QTimer>
#include <QTreeWidget>
#include <QUrl>
#include <QVBoxLayout>
#include <QWidget>

#include <algorithm>

#ifndef IPDE_SOURCE_SCRIPT
#define IPDE_SOURCE_SCRIPT "ipde_extract.py"
#endif

#ifndef IPDE_PYTHON_EXECUTABLE
#define IPDE_PYTHON_EXECUTABLE "python3"
#endif

namespace {

QString dimensionText(const QJsonObject &asset) {
    QString value = QStringLiteral("%1 × %2")
                        .arg(asset.value(QStringLiteral("width")).toInt())
                        .arg(asset.value(QStringLiteral("height")).toInt());
    const int channels = asset.value(QStringLiteral("channels")).toInt(1);
    if (channels > 1) {
        value += QStringLiteral(" × %1 ch").arg(channels);
    }
    return value;
}

QString bundledScriptPath() {
#ifdef Q_OS_MACOS
    const QDir executableDir(QCoreApplication::applicationDirPath());
    const QString bundled = executableDir.absoluteFilePath(QStringLiteral("../Resources/ipde_extract.py"));
    if (QFileInfo::exists(bundled)) {
        return QDir::cleanPath(bundled);
    }
#endif
    return QString::fromUtf8(IPDE_SOURCE_SCRIPT);
}

QString configuredPython() {
    const QString value = QString::fromUtf8(IPDE_PYTHON_EXECUTABLE);
    if (QFileInfo::exists(value)) {
        return value;
    }
    const QString found = QStandardPaths::findExecutable(QStringLiteral("python3"));
    return found.isEmpty() ? value : found;
}

class MainWindow final : public QMainWindow {
public:
    MainWindow() {
        setWindowTitle(QStringLiteral("IPDE — Precision HEIF Auxiliary Extractor"));
        resize(1040, 720);
        setAcceptDrops(true);

        auto *central = new QWidget(this);
        auto *root = new QVBoxLayout(central);
        root->setContentsMargins(20, 18, 20, 18);
        root->setSpacing(12);

        auto *title = new QLabel(QStringLiteral("Image Precision Data Extractor"), central);
        QFont titleFont = title->font();
        titleFont.setPointSize(titleFont.pointSize() + 6);
        titleFont.setBold(true);
        title->setFont(titleFont);
        root->addWidget(title);

        auto *subtitle = new QLabel(
            QStringLiteral("Extract decoded depth, gain maps, mattes, and alpha planes without normalization, tone mapping, or gamma conversion."),
            central);
        subtitle->setWordWrap(true);
        root->addWidget(subtitle);

        files_ = new QTreeWidget(central);
        files_->setColumnCount(3);
        files_->setHeaderLabels({QStringLiteral("Source / auxiliary plane"), QStringLiteral("Status / dimensions"), QStringLiteral("Precision")});
        files_->header()->setSectionResizeMode(0, QHeaderView::Stretch);
        files_->header()->setSectionResizeMode(1, QHeaderView::ResizeToContents);
        files_->header()->setSectionResizeMode(2, QHeaderView::ResizeToContents);
        files_->setSelectionMode(QAbstractItemView::ExtendedSelection);
        files_->setAlternatingRowColors(true);
        root->addWidget(files_, 1);

        auto *fileButtons = new QHBoxLayout;
        auto *add = new QPushButton(QStringLiteral("Add HEIC files…"), central);
        remove_ = new QPushButton(QStringLiteral("Remove selected"), central);
        auto *clear = new QPushButton(QStringLiteral("Clear"), central);
        inspect_ = new QPushButton(QStringLiteral("Inspect"), central);
        fileButtons->addWidget(add);
        fileButtons->addWidget(remove_);
        fileButtons->addWidget(clear);
        fileButtons->addStretch();
        fileButtons->addWidget(inspect_);
        root->addLayout(fileButtons);

        auto *outputRow = new QHBoxLayout;
        outputRow->addWidget(new QLabel(QStringLiteral("Output folder:"), central));
        output_ = new QLineEdit(central);
        output_->setPlaceholderText(QStringLiteral("Leave blank to save beside each source image"));
        auto *browse = new QPushButton(QStringLiteral("Choose…"), central);
        outputRow->addWidget(output_, 1);
        outputRow->addWidget(browse);
        root->addLayout(outputRow);

        auto *options = new QHBoxLayout;
        exactNpy_ = new QCheckBox(QStringLiteral("Write exact .npy companions"), central);
        exactNpy_->setChecked(true);
        overwrite_ = new QCheckBox(QStringLiteral("Replace existing outputs"), central);
        options->addWidget(exactNpy_);
        options->addWidget(overwrite_);
        options->addStretch();
        root->addLayout(options);

        auto *runRow = new QHBoxLayout;
        progress_ = new QProgressBar(central);
        progress_->setRange(0, 1);
        progress_->setValue(0);
        extract_ = new QPushButton(QStringLiteral("Extract verified outputs"), central);
        cancel_ = new QPushButton(QStringLiteral("Cancel"), central);
        cancel_->setEnabled(false);
        runRow->addWidget(progress_, 1);
        runRow->addWidget(extract_);
        runRow->addWidget(cancel_);
        root->addLayout(runRow);

        log_ = new QTextEdit(central);
        log_->setReadOnly(true);
        log_->setMaximumHeight(150);
        log_->setPlaceholderText(QStringLiteral("Structured extraction results appear here."));
        root->addWidget(log_);

        setCentralWidget(central);
        statusBar()->showMessage(QStringLiteral("Drop Apple HEIC portrait photos here, or choose Add HEIC files."));

        process_ = new QProcess(this);
        process_->setProcessChannelMode(QProcess::SeparateChannels);

        connect(add, &QPushButton::clicked, this, [this] {
            const QStringList paths = QFileDialog::getOpenFileNames(
                this,
                QStringLiteral("Select HEIF images"),
                QString(),
                QStringLiteral("HEIF images (*.heic *.HEIC *.heif *.HEIF *.hif *.HIF);;All files (*)"));
            addFiles(paths);
        });
        connect(remove_, &QPushButton::clicked, this, [this] {
            const auto selected = files_->selectedItems();
            QSet<QTreeWidgetItem *> roots;
            for (auto *item : selected) {
                roots.insert(item->parent() ? item->parent() : item);
            }
            for (auto *item : roots) {
                sources_.removeAll(item->data(0, Qt::UserRole).toString());
                delete item;
            }
            updateButtons();
        });
        connect(clear, &QPushButton::clicked, this, [this] {
            if (!running_) {
                sources_.clear();
                files_->clear();
                updateButtons();
            }
        });
        connect(browse, &QPushButton::clicked, this, [this] {
            const QString chosen = QFileDialog::getExistingDirectory(this, QStringLiteral("Choose output folder"), output_->text());
            if (!chosen.isEmpty()) {
                output_->setText(chosen);
            }
        });
        connect(inspect_, &QPushButton::clicked, this, [this] { beginQueue(true); });
        connect(extract_, &QPushButton::clicked, this, [this] { beginQueue(false); });
        connect(cancel_, &QPushButton::clicked, this, [this] {
            cancelled_ = true;
            queue_.clear();
            if (process_->state() != QProcess::NotRunning) {
                process_->kill();
            }
            log_->append(QStringLiteral("Cancelled by user."));
        });
        connect(process_, qOverload<int, QProcess::ExitStatus>(&QProcess::finished), this,
                [this](int exitCode, QProcess::ExitStatus exitStatus) { processFinished(exitCode, exitStatus); });
        connect(process_, &QProcess::errorOccurred, this, [this](QProcess::ProcessError error) {
            if (error == QProcess::FailedToStart) {
                const QString message = QStringLiteral("Could not start Python: %1").arg(process_->errorString());
                log_->append(message);
                if (auto *item = rootForPath(current_)) {
                    item->setText(1, QStringLiteral("Error"));
                    item->setToolTip(1, message);
                }
                ++completed_;
                progress_->setValue(completed_);
                startNext();
            }
        });

        updateButtons();
    }

protected:
    void dragEnterEvent(QDragEnterEvent *event) override {
        if (event->mimeData()->hasUrls()) {
            event->acceptProposedAction();
        }
    }

    void dropEvent(QDropEvent *event) override {
        QStringList paths;
        for (const QUrl &url : event->mimeData()->urls()) {
            const QString path = url.toLocalFile();
            if (QFileInfo(path).isFile()) {
                paths << path;
            }
        }
        addFiles(paths);
        event->acceptProposedAction();
    }

private:
    void addFiles(const QStringList &paths) {
        bool added = false;
        for (const QString &raw : paths) {
            const QString path = QFileInfo(raw).absoluteFilePath();
            if (sources_.contains(path)) {
                continue;
            }
            sources_ << path;
            auto *item = new QTreeWidgetItem(files_);
            item->setText(0, QFileInfo(path).fileName());
            item->setToolTip(0, path);
            item->setText(1, QStringLiteral("Pending inspection"));
            item->setData(0, Qt::UserRole, path);
            added = true;
        }
        updateButtons();
        if (added && !running_) {
            beginQueue(true);
        }
    }

    QTreeWidgetItem *rootForPath(const QString &path) const {
        for (int i = 0; i < files_->topLevelItemCount(); ++i) {
            auto *item = files_->topLevelItem(i);
            if (item->data(0, Qt::UserRole).toString() == path) {
                return item;
            }
        }
        return nullptr;
    }

    void beginQueue(bool inspectOnly) {
        if (running_ || sources_.isEmpty()) {
            return;
        }
        const QString python = configuredPython();
        const QString script = bundledScriptPath();
        if (!QFileInfo::exists(python)) {
            log_->append(QStringLiteral("Configured Python does not exist: %1").arg(python));
            return;
        }
        if (!QFileInfo::exists(script)) {
            log_->append(QStringLiteral("Bundled extractor does not exist: %1").arg(script));
            return;
        }
        inspectOnly_ = inspectOnly;
        cancelled_ = false;
        running_ = true;
        queue_ = sources_;
        total_ = queue_.size();
        completed_ = 0;
        progress_->setRange(0, total_);
        progress_->setValue(0);
        log_->append(inspectOnly ? QStringLiteral("Inspecting %1 source(s)…").arg(total_)
                                 : QStringLiteral("Extracting %1 source(s)…").arg(total_));
        updateButtons();
        startNext();
    }

    void startNext() {
        if (queue_.isEmpty()) {
            running_ = false;
            updateButtons();
            statusBar()->showMessage(cancelled_ ? QStringLiteral("Cancelled") : QStringLiteral("Finished"), 5000);
            return;
        }
        current_ = queue_.takeFirst();
        QStringList arguments{bundledScriptPath(), QStringLiteral("--json")};
        if (inspectOnly_) {
            arguments << QStringLiteral("--inspect");
        } else {
            if (!output_->text().trimmed().isEmpty()) {
                arguments << QStringLiteral("--output-dir") << output_->text().trimmed();
            }
            if (overwrite_->isChecked()) {
                arguments << QStringLiteral("--overwrite");
            }
            if (!exactNpy_->isChecked()) {
                arguments << QStringLiteral("--no-npy");
            }
        }
        arguments << current_;
        if (auto *item = rootForPath(current_)) {
            item->setText(1, inspectOnly_ ? QStringLiteral("Inspecting…") : QStringLiteral("Extracting…"));
        }
        statusBar()->showMessage(QStringLiteral("%1 %2").arg(inspectOnly_ ? QStringLiteral("Inspecting") : QStringLiteral("Extracting"), QFileInfo(current_).fileName()));
        process_->start(configuredPython(), arguments);
    }

    void processFinished(int exitCode, QProcess::ExitStatus exitStatus) {
        const QByteArray stdoutBytes = process_->readAllStandardOutput().trimmed();
        const QString stderrText = QString::fromUtf8(process_->readAllStandardError()).trimmed();
        QJsonParseError parseError;
        const QJsonDocument document = QJsonDocument::fromJson(stdoutBytes, &parseError);
        auto *root = rootForPath(current_);
        bool ok = exitStatus == QProcess::NormalExit && exitCode == 0 && document.isObject();
        if (document.isObject()) {
            const QJsonObject object = document.object();
            if (object.contains(QStringLiteral("error"))) {
                ok = false;
                const QString message = object.value(QStringLiteral("error")).toString();
                log_->append(QStringLiteral("%1: %2").arg(QFileInfo(current_).fileName(), message));
                if (root) {
                    root->setText(1, QStringLiteral("Error"));
                    root->setToolTip(1, message);
                }
            } else if (root) {
                while (root->childCount() > 0) {
                    delete root->takeChild(0);
                }
                const QJsonArray assets = object.value(QStringLiteral("assets")).toArray();
                root->setText(1, QStringLiteral("%1 plane(s)").arg(assets.size()));
                root->setText(2, inspectOnly_ ? QStringLiteral("Decoded inventory") : QStringLiteral("Verified outputs"));
                for (const QJsonValue &value : assets) {
                    const QJsonObject asset = value.toObject();
                    auto *child = new QTreeWidgetItem(root);
                    child->setText(0, asset.value(QStringLiteral("semantic_name")).toString());
                    child->setText(1, dimensionText(asset));
                    child->setText(
                        2,
                        QStringLiteral("%1 · source %2-bit")
                            .arg(asset.value(QStringLiteral("dtype_name")).toString())
                            .arg(asset.value(QStringLiteral("source_bit_depth")).toInt()));
                    const QString auxType = asset.value(QStringLiteral("aux_type")).toString();
                    if (!auxType.isEmpty()) {
                        child->setToolTip(0, auxType);
                    }
                }
                root->setExpanded(true);
                if (!inspectOnly_) {
                    log_->append(QStringLiteral("%1: wrote %2 verified plane(s); manifest %3")
                                     .arg(QFileInfo(current_).fileName())
                                     .arg(assets.size())
                                     .arg(object.value(QStringLiteral("manifest_path")).toString()));
                }
            }
        }
        if (!ok && !document.isObject()) {
            const QString problem = !stderrText.isEmpty()
                                        ? stderrText
                                        : QStringLiteral("Invalid extractor response: %1").arg(parseError.errorString());
            log_->append(QStringLiteral("%1: %2").arg(QFileInfo(current_).fileName(), problem));
            if (root) {
                root->setText(1, QStringLiteral("Error"));
            }
        }
        ++completed_;
        progress_->setValue(completed_);
        if (!cancelled_) {
            startNext();
        } else {
            queue_.clear();
            startNext();
        }
    }

    void updateButtons() {
        const bool hasFiles = !sources_.isEmpty();
        inspect_->setEnabled(hasFiles && !running_);
        extract_->setEnabled(hasFiles && !running_);
        remove_->setEnabled(hasFiles && !running_);
        cancel_->setEnabled(running_);
        output_->setEnabled(!running_);
        exactNpy_->setEnabled(!running_);
        overwrite_->setEnabled(!running_);
    }

    QTreeWidget *files_ = nullptr;
    QLineEdit *output_ = nullptr;
    QCheckBox *exactNpy_ = nullptr;
    QCheckBox *overwrite_ = nullptr;
    QPushButton *inspect_ = nullptr;
    QPushButton *extract_ = nullptr;
    QPushButton *remove_ = nullptr;
    QPushButton *cancel_ = nullptr;
    QProgressBar *progress_ = nullptr;
    QTextEdit *log_ = nullptr;
    QProcess *process_ = nullptr;
    QStringList sources_;
    QStringList queue_;
    QString current_;
    int total_ = 0;
    int completed_ = 0;
    bool inspectOnly_ = true;
    bool running_ = false;
    bool cancelled_ = false;
};

}  // namespace

int main(int argc, char *argv[]) {
    QApplication application(argc, argv);
    application.setApplicationName(QStringLiteral("IPDE"));
    application.setOrganizationName(QStringLiteral("OpenAI"));
    MainWindow window;
    window.show();
    if (application.arguments().contains(QStringLiteral("--smoke-test"))) {
        QTimer::singleShot(300, &application, &QCoreApplication::quit);
    }
    return application.exec();
}
