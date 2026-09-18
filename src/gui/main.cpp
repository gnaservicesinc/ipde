#include <QApplication>
#include <QCheckBox>
#include <QComboBox>
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
#include <QSettings>
#include <QMap>
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
        resize(1100, 780);
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
        files_->setHeaderLabels({QStringLiteral("Source / output — check only what you want"), QStringLiteral("Status / dimensions"), QStringLiteral("Precision")});
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
        exactNpy_->setChecked(false);
        colorMatching_ = new QCheckBox(QStringLiteral("Color Matching"), central);
        colorMatching_->setChecked(false);
        colorMatching_->setToolTip(QStringLiteral(
            "Before inference only, histogram-match each RGB channel of the non-Hero stereo view to "
            "the Hero view. Raw extracted views remain untouched. This matches marginal code-value "
            "curves; it is not an ICC conversion or a guarantee of pixelwise color equality."));
        colorHero_ = new QComboBox(central);
        colorHero_->addItem(QStringLiteral("Hero: Left view"), QStringLiteral("left"));
        colorHero_->addItem(QStringLiteral("Hero: Right view"), QStringLiteral("right"));
        colorHero_->setEnabled(false);
        colorHero_->setToolTip(QStringLiteral(
            "The Hero view is preserved unchanged; the other view receives the recorded histogram LUTs."));
        raftDevice_ = new QComboBox(central);
        raftDevice_->addItem(QStringLiteral("RAFT device: Automatic"), QStringLiteral("auto"));
        raftDevice_->addItem(QStringLiteral("RAFT device: Apple Metal"), QStringLiteral("mps"));
        raftDevice_->addItem(QStringLiteral("RAFT device: CPU"), QStringLiteral("cpu"));
        raftDevice_->addItem(QStringLiteral("RAFT device: CUDA"), QStringLiteral("cuda"));
        raftDevice_->setToolTip(QStringLiteral(
            "Automatic prefers Apple Metal on this Mac and falls back to CPU only when Metal is unavailable."));
        overwrite_ = new QCheckBox(QStringLiteral("Replace existing outputs"), central);
        options->addWidget(exactNpy_);

        options->addWidget(overwrite_);
        options->addStretch();
        root->addLayout(options);

        auto *spatialOptions = new QHBoxLayout;
        spatialOptions->addWidget(new QLabel(QStringLiteral("Spatial Photo:"), central));
        spatialOptions->addWidget(colorMatching_);
        spatialOptions->addWidget(colorHero_);
        spatialOptions->addWidget(raftDevice_);
        spatialOptions->addStretch();
        root->addLayout(spatialOptions);

        auto addPathRow = [this, root, central](const QString &label, const QString &key,
                                                bool directory) {
            auto *row = new QHBoxLayout;
            row->addWidget(new QLabel(label, central));
            auto *edit = new QLineEdit(QSettings().value(key).toString(), central);
            edit->setPlaceholderText(QStringLiteral("Automatic lookup (or choose a path)"));
            auto *choose = new QPushButton(QStringLiteral("Choose…"), central);
            row->addWidget(edit, 1);
            row->addWidget(choose);
            root->addLayout(row);
            connect(edit, &QLineEdit::textChanged, this, [key](const QString &text) {
                QSettings().setValue(key, text);
            });
            connect(choose, &QPushButton::clicked, this, [this, edit, directory] {
                if (running_) return;
                const QString chosen = directory
                    ? QFileDialog::getExistingDirectory(this, QStringLiteral("Choose RAFT-Stereo source folder"), edit->text())
                    : QFileDialog::getOpenFileName(this, QStringLiteral("Choose RAFT-Stereo model"), edit->text(),
                          QStringLiteral("Model checkpoints (*.pth *.pt *.zip);;All files (*)"));
                if (!chosen.isEmpty()) edit->setText(chosen);
            });
            return edit;
        };
        raftModel_ = addPathRow(QStringLiteral("RAFT model:"), QStringLiteral("raft/model"), false);
        raftRoot_ = addPathRow(QStringLiteral("RAFT source folder:"), QStringLiteral("raft/root"), true);
        auto *memberRow = new QHBoxLayout;
        memberRow->addWidget(new QLabel(QStringLiteral("Model inside ZIP:"), central));
        raftMember_ = new QLineEdit(QSettings().value(QStringLiteral("raft/member")).toString(), central);
        raftMember_->setPlaceholderText(QStringLiteral("raftstereo-middlebury.pth (only for ZIP models)"));
        memberRow->addWidget(raftMember_);
        root->addLayout(memberRow);
        connect(raftMember_, &QLineEdit::textChanged, this, [](const QString &text) {
            QSettings().setValue(QStringLiteral("raft/member"), text);
        });
        auto *help = new QLabel(QStringLiteral(
            "Check individual outputs, then Export checked. Or select one row and click Export this map. "
            "For a 0–1 height input choose RAFT height — 0–1 displacement. Raw pixel disparity can look white "
            "in a 0–1 viewer; unmatched classical pixels remain NaN."), central);
        help->setWordWrap(true);
        root->addWidget(help);


        auto *runRow = new QHBoxLayout;
        progress_ = new QProgressBar(central);
        progress_->setRange(0, 1);
        progress_->setValue(0);
        extract_ = new QPushButton(QStringLiteral("Export checked"), central);
        exportOne_ = new QPushButton(QStringLiteral("Export this map"), central);
        runRow->addWidget(exportOne_);
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
                while (item->parent()) item = item->parent();
                roots.insert(item);
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
            if (running_) return;
            const QString chosen = QFileDialog::getExistingDirectory(this, QStringLiteral("Choose output folder"), output_->text());
            if (!chosen.isEmpty()) {
                output_->setText(chosen);
            }
        });
        connect(inspect_, &QPushButton::clicked, this, [this] { beginQueue(true); });
        connect(extract_, &QPushButton::clicked, this, [this] { beginQueue(false); });
        connect(exportOne_, &QPushButton::clicked, this, [this] {
            auto *item = files_->currentItem();
            if (!item || item->data(0, Qt::UserRole + 1).toString().isEmpty()) return;
            singleSource_ = item->parent()->data(0, Qt::UserRole).toString();
            singleProduct_ = item->data(0, Qt::UserRole + 1).toString();
            beginQueue(false);
            singleSource_.clear();
            singleProduct_.clear();
        });
        connect(files_, &QTreeWidget::itemSelectionChanged, this, [this] { updateButtons(); });
        connect(files_, &QTreeWidget::itemChanged, this, [this] { updateButtons(); });
        connect(colorMatching_, &QCheckBox::toggled, this, [this](bool checked) {
            colorHero_->setEnabled(checked && !running_);
        });
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
        if (running_) return;
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
        selectedProducts_.clear();
        if (!inspectOnly) {
            if (!singleProduct_.isEmpty()) {
                selectedProducts_[singleSource_] = {singleProduct_};
            } else {
                for (int i = 0; i < files_->topLevelItemCount(); ++i) {
                    auto *source = files_->topLevelItem(i);
                    QStringList products;
                    for (int j = 0; j < source->childCount(); ++j) {
                        auto *child = source->child(j);
                        if (child->checkState(0) == Qt::Checked)
                            products << child->data(0, Qt::UserRole + 1).toString();
                    }
                    if (!products.isEmpty()) selectedProducts_[source->data(0, Qt::UserRole).toString()] = products;
                }
            }
            if (selectedProducts_.isEmpty()) {
                log_->append(QStringLiteral("Check an output or select a row and use Export this map."));
                return;
            }
        }
        inspectOnly_ = inspectOnly;
        cancelled_ = false;
        running_ = true;
        queue_ = inspectOnly ? sources_ : selectedProducts_.keys();
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
            for (const auto &id : selectedProducts_.value(current_))
                arguments << QStringLiteral("--select") << id;
            arguments << QStringLiteral("--raft-device") << raftDevice_->currentData().toString();
            if (colorMatching_->isChecked()) {
                arguments << QStringLiteral("--color-matching") << QStringLiteral("--color-hero")
                          << colorHero_->currentData().toString();
            }
            if (!raftModel_->text().trimmed().isEmpty())
                arguments << QStringLiteral("--raft-model") << raftModel_->text().trimmed();
            if (!raftRoot_->text().trimmed().isEmpty())
                arguments << QStringLiteral("--raft-root") << raftRoot_->text().trimmed();
            if (!raftMember_->text().trimmed().isEmpty())
                arguments << QStringLiteral("--raft-model-member") << raftMember_->text().trimmed();

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
                QSet<QString> checked;
                const QString currentProduct = files_->currentItem()
                    ? files_->currentItem()->data(0, Qt::UserRole + 1).toString() : QString();
                for (int j = 0; j < root->childCount(); ++j) {
                    if (root->child(j)->checkState(0) == Qt::Checked)
                        checked.insert(root->child(j)->data(0, Qt::UserRole + 1).toString());
                }
                while (root->childCount() > 0) {
                    delete root->takeChild(0);
                }
                const QJsonArray assets = object.value(QStringLiteral("assets")).toArray();
                const QJsonObject source = object.value(QStringLiteral("source")).toObject();
                const QJsonObject spatial = source.value(QStringLiteral("spatial_photo")).toObject();
                int outputCount = 0;
                root->setText(
                    1,
                    spatial.isEmpty() ? QStringLiteral("%1 plane(s)").arg(assets.size())
                                      : QStringLiteral("Spatial Photo · %1 plane(s)").arg(assets.size()));
                root->setText(2, inspectOnly_ ? QStringLiteral("Decoded inventory") : QStringLiteral("Verified outputs"));
                for (const QJsonValue &value : assets) {
                    const auto outputs = value.toObject().value(QStringLiteral("outputs")).toArray();
                    outputCount += outputs.size();
                    for (const auto &entry : outputs)
                        log_->append(entry.toObject().value(QStringLiteral("path")).toString().toHtmlEscaped());
                }
                const auto products = object.value(QStringLiteral("available_products")).toArray();
                for (const auto &entry : products) {
                    const auto product = entry.toObject();
                    const auto id = product.value(QStringLiteral("id")).toString();
                    auto *child = new QTreeWidgetItem(root);
                    child->setText(0, product.value(QStringLiteral("name")).toString());
                    child->setText(1, dimensionText(product));
                    child->setText(2, product.value(QStringLiteral("precision")).toString());
                    child->setData(0, Qt::UserRole + 1, id);
                    child->setFlags(child->flags() | Qt::ItemIsUserCheckable);
                    child->setCheckState(0, checked.contains(id) ? Qt::Checked : Qt::Unchecked);
                    if (id == currentProduct) files_->setCurrentItem(child);
                }
                for (const auto &warning : object.value(QStringLiteral("warnings")).toArray())
                    log_->append(warning.toString().toHtmlEscaped());
                root->setExpanded(true);
                if (!spatial.isEmpty()) {
                    log_->append(
                        QStringLiteral("%1: Spatial Photo — left %2, right %3, baseline %4 mm, disparity adjustment %5%")
                            .arg(QFileInfo(current_).fileName())
                            .arg(spatial.value(QStringLiteral("left_image_index")).toInt())
                            .arg(spatial.value(QStringLiteral("right_image_index")).toInt())
                            .arg(spatial.value(QStringLiteral("baseline_millimeters")).toDouble(), 0, 'f', 6)
                            .arg(spatial.value(QStringLiteral("disparity_adjustment_fraction_of_width")).toDouble() * 100.0, 0, 'f', 4));
                }
                if (!inspectOnly_) {
                    log_->append(QStringLiteral("%1: wrote %2 verified file(s); manifest %3")
                                     .arg(QFileInfo(current_).fileName())
                                     .arg(outputCount)
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
        colorMatching_->setEnabled(!running_);
        colorHero_->setEnabled(!running_ && colorMatching_->isChecked());
        raftDevice_->setEnabled(!running_);
        raftModel_->setEnabled(!running_);
        raftRoot_->setEnabled(!running_);
        raftMember_->setEnabled(!running_);
        files_->setEnabled(!running_);
        const auto *item = files_->currentItem();
        exportOne_->setEnabled(!running_ && item && !item->data(0, Qt::UserRole + 1).toString().isEmpty());
        overwrite_->setEnabled(!running_);
    }

    QTreeWidget *files_ = nullptr;
    QLineEdit *output_ = nullptr;
    QLineEdit *raftModel_ = nullptr;
    QLineEdit *raftRoot_ = nullptr;
    QLineEdit *raftMember_ = nullptr;
    QPushButton *exportOne_ = nullptr;
    QMap<QString, QStringList> selectedProducts_;
    QString singleSource_;
    QString singleProduct_;
    QCheckBox *exactNpy_ = nullptr;
    QCheckBox *colorMatching_ = nullptr;
    QComboBox *colorHero_ = nullptr;
    QComboBox *raftDevice_ = nullptr;
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
