import 'dart:convert';
import 'package:flutter/material.dart';
import 'package:file_picker/file_picker.dart';
import 'package:open_file/open_file.dart';
import 'package:permission_handler/permission_handler.dart';
import 'package:http/http.dart' as http;

void main() => runApp(MyApp());

class MyApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Resume Matcher',
      theme: ThemeData(
        primarySwatch: Colors.blue,
      ),
      home: MyHomePage(),
    );
  }
}

class MyHomePage extends StatefulWidget {
  @override
  _MyHomePageState createState() => _MyHomePageState();
}

class _MyHomePageState extends State<MyHomePage> {
  List<String> resumeFiles = [];
  List<String> jobDescriptionFiles = [];

  Future<void> _pickFiles(bool isResume) async {
    FilePickerResult? result = await FilePicker.platform.pickFiles(
      type: FileType.custom,
      allowMultiple: true,
      allowedExtensions: ['pdf', 'docx'],
    );

    if (result != null) {
      setState(() {
        if (isResume) {
          resumeFiles.addAll(result.paths!.cast<String>());
        } else {
          jobDescriptionFiles.addAll(result.paths!.cast<String>());
        }
      });
    }
  }

  void _removeFile(bool isResume, String filePath) {
    setState(() {
      if (isResume) {
        resumeFiles.remove(filePath);
      } else {
        jobDescriptionFiles.remove(filePath);
      }
    });
  }

  Future<void> requestPermission() async {
    final permission = Permission.storage;

    if (await permission.isDenied) {
      await permission.request();
    }
  }

  void _openFile(String filePath) async {
    // Cek izin untuk membuka file
    var status = await Permission.storage.status;

    if (status.isGranted) {
      OpenFile.open(filePath);
    } else if (status.isPermanentlyDenied) {
      // Jika izin secara permanen ditolak, beri tahu pengguna untuk pergi ke pengaturan dan mengaktifkannya
      openAppSettings();
    } else {
      // Jika izin belum diberikan, minta izin
      if (await Permission.storage.request().isGranted) {
        OpenFile.open(filePath);
      }
    }
  }

  Future<void> _processFiles() async {
    final apiUrl = 'http://45.9.191.89:5000/upload';

    var request = http.MultipartRequest('POST', Uri.parse(apiUrl));

    // Add resume files
    for (String resumePath in resumeFiles) {
      request.files
          .add(await http.MultipartFile.fromPath('resumes', resumePath));
    }

    // Add job description files
    for (String jobDescriptionPath in jobDescriptionFiles) {
      request.files.add(await http.MultipartFile.fromPath(
          'job_descriptions', jobDescriptionPath));
    }

    try {
      final response = await request.send();

      if (response.statusCode == 200) {
        print('Files uploaded');
        final result = jsonDecode(await response.stream.bytesToString());
        _showResultDialog(result);
      } else {
        print('Error: ${response.reasonPhrase}');
      }
    } catch (e) {
      print('Error: $e');
    }
  }

  void _showResultDialog(Map<String, dynamic> result) {
    showDialog(
      context: context,
      builder: (BuildContext context) {
        return AlertDialog(
          title: Text('Hasil Pencocokan'),
          content: SingleChildScrollView(
            child: Column(
              children: [
                Text('${result['top_resumes']}'),
              ],
            ),
          ),
          actions: [
            TextButton(
              onPressed: () {
                Navigator.of(context).pop();
              },
              child: Text('Tutup'),
            ),
          ],
        );
      },
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('Resume Matcher'),
      ),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: SingleChildScrollView(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              ElevatedButton(
                onPressed: () => _pickFiles(false),
                child: Text('Pilih Job Description Files'),
              ),
              SizedBox(height: 32),
              ElevatedButton(
                onPressed: () => _pickFiles(true),
                child: Text('Pilih Resume Files'),
              ),
              SizedBox(height: 16),
              Text('Job Description Files: ${jobDescriptionFiles.length}'),
              Column(
                children: jobDescriptionFiles
                    .map(
                      (file) => Row(
                        children: [
                          Expanded(
                            child: Text(
                              file.split('/').last,
                              // Menampilkan nama file saja
                              maxLines: 1,
                              overflow: TextOverflow.ellipsis,
                            ),
                          ),
                          IconButton(
                            icon: Icon(Icons.delete),
                            onPressed: () => _removeFile(false, file),
                          ),
                          IconButton(
                            icon: Icon(
                                Icons.open_in_new), // Ikon untuk membuka file
                            onPressed: () =>
                                _openFile(file), // Panggil metode _openFile
                          ),
                        ],
                      ),
                    )
                    .toList(),
              ),
              SizedBox(height: 32),
              Text('Resume Files: ${resumeFiles.length}'),
              Column(
                children: resumeFiles
                    .map(
                      (file) => Row(
                        children: [
                          Expanded(
                            child: Text(
                              file.split('/').last,
                              // Menampilkan nama file saja
                              maxLines: 1,
                              overflow: TextOverflow.ellipsis,
                            ),
                          ),
                          IconButton(
                            icon: Icon(Icons.delete),
                            onPressed: () => _removeFile(true, file),
                          ),
                          IconButton(
                            icon: Icon(
                                Icons.open_in_new), // Ikon untuk membuka file
                            onPressed: () =>
                                _openFile(file), // Panggil metode _openFile
                          ),
                        ],
                      ),
                    )
                    .toList(),
              ),
              SizedBox(height: 16),
              ElevatedButton(
                onPressed: _processFiles,
                child: Text('Proses dan Tampilkan Hasil'),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
